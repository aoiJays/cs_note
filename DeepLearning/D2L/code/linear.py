import torch
from torch import nn
import numpy as np
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = torch.tensor([4.2])

# 创建数据集

# d2l的辅助函数 生成带有噪声的线性回归数据
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
dataset = data.TensorDataset( *(features, labels ) )
'''
* 解包运算符
print( *(1,1), (1,1))
1 1 (1, 1)
'''

batch_size = 10
dataLoder = data.DataLoader(dataset, batch_size, shuffle=True)

# 模型初始化
net = nn.Sequential( nn.Linear(2, 1) )
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):

    for X,y in dataLoder:
        l = loss(y, net(X))
        trainer.zero_grad() # 清空梯度
        l.backward()
        trainer.step()

    l  = loss(net(features), labels)
    print( f'epoch {epoch + 1} , loss {l:f}')

print(net[0].weight.data)
print(net[0].bias.data)