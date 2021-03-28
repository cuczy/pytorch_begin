import torch
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

'''
主要讲述了dropout的使用，主要是两点，一是加dropout的话直接在网络层中加，二是预测时要加上.eval()进入预测模式，要不然还在使用dropout
'''

N_SAMPLES = 20
N_HIDDEN = 300
#训练数据和测试数据基本上是都是在一个方向上，当模型在训练集上loss很小时会发现在test上loss很大
# training data
x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
y = x + 0.3*torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))

# test data
test_x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
test_y = test_x + 0.3*torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))

# # show data
# plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.5, label='train')
# plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.5, label='test')
# plt.legend(loc='upper left')
# plt.ylim((-2.5, 2.5))
# plt.show()

net_overfitting = torch.nn.Sequential(
    torch.nn.Linear(1, N_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, N_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, 1),
)

net_drop = torch.nn.Sequential(
    torch.nn.Linear(1, N_HIDDEN),
    torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, N_HIDDEN),
    torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, 1),
)

optimizer_over = torch.optim.Adam(net_overfitting.parameters(), lr=0.01)
optimizer_drop = torch.optim.Adam(net_drop.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()

plt.ion()
for t in range(500):
    pred_over = net_overfitting(x)
    pred_drop = net_drop(x)
    loss_over = loss_func(pred_over, y)
    loss_drop = loss_func(pred_drop, y)

    optimizer_over.zero_grad()
    optimizer_drop.zero_grad()
    loss_over.backward()
    loss_drop.backward()
    optimizer_over.step()
    optimizer_drop.step()

    if t % 10 == 0:
        net_overfitting.eval()
        net_drop.eval()#关键的一步，因为这个网络在训练时使用了dropout，但是现在要用去预测，要使用全部完整的网络，加.eval()为预测模式
        #预测一下
        test_pred_over = net_overfitting(test_x)
        test_pred_drop = net_drop(test_x)

        # plotting
        plt.cla()

        plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.3, label='train')
        plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.3, label='test')
        plt.plot(test_x.data.numpy(), test_pred_over.data.numpy(), 'r-', lw=3, label='overfitting')
        plt.plot(test_x.data.numpy(), test_pred_drop.data.numpy(), 'b--', lw=3, label='dropout(50%)')
        plt.text(0, -1.2, 'overfitting loss=%.4f' % loss_func(test_pred_over, test_y).data.numpy(),
                 fontdict={'size': 20, 'color': 'red'})
        plt.text(0, -1.5, 'dropout loss=%.4f' % loss_func(test_pred_drop, test_y).data.numpy(),
                 fontdict={'size': 20, 'color': 'blue'})
        plt.legend(loc='upper left');
        plt.ylim((-2.5, 2.5));
        plt.pause(0.1)

        net_overfitting.train()#预测完了再改回来训练模式
        net_drop.train()
plt.ioff()
plt.show()