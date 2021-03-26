import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy
'''
核心的代码也就那么几行，这里因为说RNN效果不好就使用LSTM了，构建LSTM也是使用现有的模块，设置一点参数就行
需要注意的是前向传播，rnn的返回有两个；在最后做分类预测时只需要提出来最后一步的输出结果
'''
EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28  #考虑多少个时间点
INPUT_SIZE = 28 #每个时间点上给多少个数据   （在这块是相当于图片是28*28的，从上到下一共28步， 每一步横着28的宽度）
LR = 0.01
DOWNLOAD_MNIST = True

train_data = dsets.MNIST(root='./mnist', train=True, transform=transforms.ToTensor(), download=DOWNLOAD_MNIST)#transform的作用是把图片压缩到0，1之间
train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = True)

test_data = dsets.MNIST(root = './mnist/', train = False, transform=transforms.ToTensor)
test_x = Variable(test_data.test_data, volatile=True).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels.numpy().squeeze()[:2000]

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,#贯穿整个的主线，多的话运行慢
            batch_first=True, #是否把batch_size放在第一个维度
        )
        self.out = nn.Linear(64, 10)
    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)#其中(h_n, h_c)是hidden_state的其中两个部分，hidden_state是中间产物，结合input一起计算得到r_out,一直迭代
        #但是我们目前做分类不需要中间的，只要最后的输出， None是因为这里第一个没有hidden_state的输入，后面回归会详细讲到
        out = self.out(r_out[:, -1, :]) #只取最后一步的输出(batch, time_step, input)
        return out
rnn = RNN()

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x.view(-1, 28, 28))#把输入形状调整到(batch, time_step, input)，-1表示自动适配
        b_y = Variable(y)
        output = rnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()#梯度清零
        loss.backward()
        optimizer.step()

        if step % 50 ==0:
            test_output = rnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = sum(pred_y == test_y)/test_y.size
            print('Epoch:', epoch, '|train_loss:%.4f'%loss.data, '|test_accuracy:%.2f'%accuracy)

#看一下前十个有没有预测对
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction_number')
print(test_y[:10], 'real_number')
