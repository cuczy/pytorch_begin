import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(1)

TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.02

#根据sin预测cos
steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
x_np = np.sin(steps)
y_np = np.cos(steps)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(32, 1)
    def forward(self, x, h_state):
        #x (batch, time_step, input_size)
        #h_state(n_layers, batch, hidden_size)
        #r_out (batch, time_step, hidden_size)

        r_out, h_state = self.rnn(x, h_state)#这里有h_state是因为传统的RNN是需要结合输入与隐藏层记忆一起计算
        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, -1, :]))#取出来rrn的输出，中间的每个时间点到最后一层过一下，然后把结果存在outs中
        return torch.stack(outs, dim=1), h_state
rnn = RNN()

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.MSELoss()

h_state = None#这是为了给下面预测最开始时需要有个数传进去，后面自己就开始迭代，所以第一个就传个空进去
for step in range(60):
    start, end = step * np.pi, (step + 1) * np.pi#截取了起点和终点
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)#取TIME_STEP这么多步的点
    x_np = np.sin(steps)
    y_np = np.cos(steps)

    x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis]))#之前是一维的数据，现在增加维度变成(batch, time_step, input_size)
    y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis]))

    predition, h_state = rnn(x, h_state)
    h_state = Variable(h_state.data)#开始迭代传进去之前要用variable包一下
    loss = loss_func(predition, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    '''
    后面就是画图的代码，视频上没看到
    '''
