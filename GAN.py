import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn

torch.manual_seed(1)

BATCH_SIZE = 64
LR_G = 0.0001
LR_D = 0.0001
N_IDEAS = 5#假设有5个人灵感，这个后面是用在网络的输入
ART_COMPONENTS = 15
#开始学习画画，目标是能在两个曲线（代表合格的范围）之间画出曲线
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])

def artist_works():#这是真画家的成果
    #制造出两条曲线，代表合格标准
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    paintings = a * np.power(PAINT_POINTS, 2) + (a - 1)
    #转换成tensor的形式，然后用Variable包一下
    paintings = torch.from_numpy(paintings).float()
    return Variable(paintings)

G = nn.Sequential(
    nn.Linear(N_IDEAS, 128),
    nn.ReLU(),
    nn.Linear(128, ART_COMPONENTS),

)

D = nn.Sequential(
    nn.Linear(ART_COMPONENTS, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),#因为要判断是否是好的画，所以用百分比的形式看他多大概率是好画
)

opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

plt.ion()
for step in range(10000):
    artist_painting = artist_works()#这是真画家的作品
    G_idea = Variable(torch.randn(BATCH_SIZE, N_IDEAS))#随机产生这样的维度的输入灵感
    G_paintings = G(G_idea)#这是新手画的

    prob_artist0 = D(artist_painting)
    prob_artist1 = D(G_paintings)
    #目标是让真是画家的画在判别器中得到高分，新手画得到低分，这才是判别器的能力(因为只有最小化，所以加负号)
    D_loss = -torch.mean(torch.log(prob_artist0) + torch.log(1 - prob_artist1))
    #生成器目标是让新手画的图尽可能让判别器识别不出来而打高分,这个后面也是要最小户，所以相当于最大化pro1
    G_loss = torch.mean(torch.log(1 - prob_artist1))

    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)#因为有两次反向传播，这个的意思的这个完了之后会保留参数，在进行下面的不会有影响
    opt_D.step()

    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

    # if step % 100 == 0:
    #     print('Epoch:', step, '|D_loss:%.4f' % D_loss.data, '|G_loss:%.4f'%G_loss.data)
    if step % 50 == 0:  # plotting
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting', )
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(),
                 fontdict={'size': 13})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
        plt.ylim((0, 3));
        plt.legend(loc='upper right', fontsize=10);
        plt.draw();
        plt.pause(0.01)

plt.ioff()
plt.show()