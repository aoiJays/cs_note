# Pytorch入门

[TOC]

## 环境配置

```bash
conda create -n pytorch-learn python=3.8
```

查看cuda版本：

```bash
nvcc --version
```

根据cuda版本进行选择：[Pytorch本地安装](https://pytorch.org/get-started/locally/)

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

安装检测

```cpp
(pytorch-learn) aoijays@aoijays-ubuntu:~/Desktop/note$ python                                       
Python 3.8.19 (default, Mar 20 2024, 19:58:24)                                                      
[GCC 11.2.0] :: Anaconda, Inc. on linux                                                             
Type "help", "copyright", "credits" or "license" for more information.                              
>>> import torch                                                                                     
>>> torch.cuda.is_available()
True        
```

此时说明安装成功



## 前言

### 法宝函数

```python
dir( ... ) # 显示下属对象列表
help( ... ) # 显示当前对象的说明
## 但我更喜欢加两个问号??
```

![image-20240529020547606](./Pytorch_小土堆.assets/image-20240529020547606.png)

### Dataset

 以[蜜蜂蚂蚁数据集](https://www.kaggle.com/datasets/ajayrana/hymenoptera-data)进行说明

其文件目录：

```bash
│   └── hymenoptera_data
│       ├── train
│       │   ├── ants
│       │   └── bees
│       └── val
│           ├── ants
│           └── bees
```

子文件夹小附有若干张jpg

接下来我们需要使用torch的Dataset去加载数据集



>  |  An abstract class representing a :class:`Dataset`.
>  |  
>  |  All datasets that represent <u>**a map from keys to data samples**</u> should subclass
>  |  it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
>  |  data sample for a given key. Subclasses could also optionally overwrite
>  |  :meth:`__len__`, which is expected to return the size of the dataset by many
>  |  :class:`~torch.utils.data.Sampler` implementations and the default options
>  |  of :class:`~torch.utils.data.DataLoader`. Subclasses could also
>  |  optionally implement :meth:`__getitems__`, for speedup batched samples
>  |  loading. This method accepts list of indices of samples of batch and returns
>  |  list of samples.

省流：

- 需要继承
- 必须重写`__getitem__`
- 可以重写`__len__`

```python
from torch.utils.data import Dataset
from PIL import Image
import os

class MyDataet(Dataset):

    def __init__(self, root_dir, label):
        
        # 自定义 怎么方便怎么来
        self.root_dir = root_dir # 记录数据地址以及对应的标签
        self.label = label
        self.imglist = os.listdir(self.root_dir) # 以列表形式展示文件夹内所有文件
    
    def __getitem__(self, idx):
        img_name= self.imglist[idx]
        img_path = os.path.join( self.root_dir, img_name )
        
        img = Image.open(img_path) # 打开图片
        label = self.label

        return img, label # 返回数据与标签
    
    def __len__(self):
        return len(self.imglist)
    
train_ants = MyDataet('../Dataset/hymenoptera_data/train/ants', 'ants')  
train_bees = MyDataet('../Dataset/hymenoptera_data/train/bees', 'bees')  

train_data = train_ants + train_bees # 拼接数据集
print( train_ants.__len__(),  train_bees.__len__(), train_data.__len__())
print(train_data[123], train_data[124], sep='\n')
# 124 121 245
# (<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x375 at 0x775FB149ADF0>, 'ants')
# (<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=311x387 at 0x775FB0EB29A0>, 'bees')
```

写法非常自由，你只要保证重写的函数返回正确结果即可



### Tensorboard

```bash
conda install tensorboard
```
#### Scalars

绘制一些图表，观察训练的loss变化



```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("logs") # 创建对象 在logs文件夹下保存文件
for i in range(100):
    writer.add_scalar("y = x", i, i) # 图像名 y值 x值

for i in range(100):
    writer.add_scalar("y = x^2", i*i, i) # 图像名 y值 x值

writer.close()

```

在终端中：

```bash
tensorboard --logdir=logs --port=6007 # 默认6006
```

![image-20240529030116600](./Pytorch_小土堆.assets/image-20240529030116600.png)

#### Images

可视化实际的训练效果，上传图片进行展示

```python
img = Image.open(
    '../Dataset/hymenoptera_data/train/ants/0013035.jpg'
)
print(type(img)) # <class 'PIL.JpegImagePlugin.JpegImageFile'>

# 不支持此类型 需要转化为numpy对象
img_array = np.array(img)
print(img_array.shape) # (512, 768, 3)

# 高 宽 通道数 -> HWC
# 标题 添加的图像 步 格式
writer.add_image("test", img_array, 1, dataformats="HWC")

# 我们可以第1步展示一张图，第2步展示一张图……
# 即可观察随着训练步数，图片发生变化
```



![image-20240529031540127](./Pytorch_小土堆.assets/image-20240529031540127.png)



### Transforms

 留个印象就好，能通过这个方法对数据、图片等进行互相转换、变化

常见的有：

- ToTensor：转化为Tensor数据
- Normalize：归一化
- Resize：对图片数据进行缩放
- Compose：用列表记录多个变化，进行一次性操作

可能用到的时候查一下就行

适合对多个数据同时进行相同的处理



### Torchvision数据集的下载与使用

```python
import torchvision
train_set = torchvision.datasets.CIFAR10(root='../Dataset', train=True, download=True) # 训练集
test_set = torchvision.datasets.CIFAR10(root='../Dataset', train=False, download=True) # 测试集

print(train_set[0])
print(train_set.classes)
img, target = test_set[0]
print(img, target)

'''
(<PIL.Image.Image image mode=RGB size=32x32 at 0x7202D3007670>, 6)
['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
<PIL.Image.Image image mode=RGB size=32x32 at 0x7202D3007310> 3
'''
```

此时我们得到的数据集都是`(PIL对象，标签索引)`

我们可以使用Transform统一把图片转化为Tensor，方便Pytorch使用

最简单的方法：

```python
dataset_transform = torchvision.transforms.Compose([ # 封装所有需要进行的转换列表
    torchvision.transforms.ToTensor()
])

# 在加载数据时直接进行参数化修改
train_set = torchvision.datasets.CIFAR10(root='../Dataset', train=True, transform=dataset_transform, download=True) # 训练集
test_set = torchvision.datasets.CIFAR10(root='../Dataset', train=False, transform=dataset_transform, download=True) # 测试集

```



### DataLoader

```python
import torchvision
from torch.utils.data import DataLoader

test_set = torchvision.datasets.CIFAR10(root='../Dataset', train=False, transform=torchvision.transforms.ToTensor(), download=True) # 测试集

# 除了官方数据集 也可以选择之前自己实例化的Dataset
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, drop_last=False)

for data in test_loader: # 按batch_size数量进行遍历
    imgs, targets = data
    print(targets)
```



## 网络

 

```python
import torch
from torch import nn

# 需要从nn.Module进行继承
class Mynn(nn.Module):
    
	def __init__(self):
		super().__init__() # 调用父类的初始化函数
   
	def forward(self, input): # 前向传播
		output = input + 1
		return output
        

mynn = Mynn()
x = torch.tensor(1.0)
output = mynn(x)
print(output)
```

 

### 卷积层

对于一张H*W的RGB图片，其通道数channel为3

我们使用多少个卷积核，就会产生多少个out_channels

```python
class Mycnn(nn.Module):
    
	def __init__(self):
		super().__init__() 
        # 输入3通道 用6个3*3的卷积核得到6通道输出 步长为1 不填充
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

	def forward(self, input): # 前向传播
		output = self.conv1(input)
		return output
        
mycnn = Mycnn()
x = torch.FloatTensor([
	[[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]],
	[[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]],
	[[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]
])

mycnn(x)
```



### 池化层

 卷积后的图像依旧比较大，可以通过池化层进行压缩

>   避免过拟合、去除冗余



```python
# 池化

class Mycnn(nn.Module):
    
	def __init__(self):
		super().__init__() 
		# ceil_mode 若有无法整除kernel_size 多余的部分按照ceil_mode决定是否保留
		self.maxpool1 = nn.MaxPool2d( kernel_size=3, ceil_mode=True)

	def forward(self, input): # 前向传播
		output = self.maxpool1(input)
		return output
        
mycnn = Mycnn()
x = torch.tensor([
	[[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]],
	[[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]],
	[[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]
], dtype = torch.float32)

# x.shape
mycnn(x)
```



我们可以喂入图片，就可以得到压缩画质版本的输出



### 激活层

```python
# ReLU
import torch
from torch import nn

class Mynn(nn.Module):
    
	def __init__(self):
		super().__init__()
		self.ReLU = nn.ReLU()
	
	def forward(self, input): # 前向传播
		output = self.ReLU(input)
		return output
        
mynn = Mynn()

input = torch.tensor([
	[1., -0.5],
	[-1., 3]
])

input = torch.reshape(input, (-1, 1, 2, 2))

print(input)
output = mynn(input)
print(output)
```

### 其他

-   正则化层
-   线性层……

### Sequential

我们试图构建一个较大的网络对CIFAR10数据集进行推理

![image-20240530152550478](./Pytorch_小土堆.assets/image-20240530152550478.png)

```python
# Sequential
import torch
from torch import nn

class Mynn(nn.Module):

    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(), # 将tensor张成一维张量
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.sequential(x)
        return x

mynn = Mynn()
print(mynn)

# 模拟一个batch_size = 64中32*32的3通道数据集
input = torch.ones( (64, 3, 32, 32) )
output = mynn(input)
print(output.shape)
```

## 损失函数

```python
import torch
from torch.nn import *

inputs = torch.tensor([1,2,3], dtype=torch.float32).reshape([-1,1,1,3])
targets = torch.tensor([1,2,5], dtype=torch.float32).reshape([-1,1,1,3])

loss1 = L1Loss(reduction='sum')
loss2 = L1Loss(reduction='mean')

print( loss1(inputs, targets) )
print( loss2(inputs, targets) )

loss3 = MSELoss()
print( loss3(inputs, targets) )

import torch
import torch.nn as nn

# 交叉熵损失函数
# 假设 outputs 是模型的最后一层输出，shape 为 (batch_size, num_classes)，targets 是 ground truth labels
outputs = torch.randn(10, 4)  # 对于4分类问题的10个样本的不归一化的预测值
targets = torch.randint(0, 4, (10,))  # 对应的真实类别

print(outputs)
print(targets)

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(outputs, targets)
print(loss.item())

```



## 反向传播

```python
# 构建网络
import torch
from torch import nn

class Mynn(nn.Module):

    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(), # 将tensor张成一维张量
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.sequential(x)
        return x

mynn = Mynn()
print(mynn)

# 模拟一个batch_size = 64中32*32的3通道数据集
input = torch.ones( (64, 3, 32, 32) )
output = mynn(input)
print(output.shape)

# -------------------------------------------
# 构建数据集
import torchvision
from torch.utils.data import DataLoader

test_set = torchvision.datasets.CIFAR10(root='../Dataset', train=False, transform=torchvision.transforms.ToTensor(), download=True) # 测试集

test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, drop_last=False)

# -------------------------------------------
# 反向传播

loss_fn = nn.CrossEntropyLoss()

for data in test_loader: # 按batch_size数量进行遍历
    imgs, targets = data
    outputs = mynn(imgs)
    
    # 计算得到损失函数
    loss = loss_fn(outputs, targets)
    loss.backward() # 反向传播 沿计算图得到所有参数的梯度

```



## 优化器

```python

loss_fn = nn.CrossEntropyLoss()

loss_sum = 0
for data in test_loader: # 按batch_size数量进行遍历
    
    # 正向推理
    imgs, targets = data
    outputs = mynn(imgs)
    
    # 计算得到损失函数
    loss = loss_fn(outputs, targets)

    optim.zero_grad() # 将所有参数进行梯度清零
    loss.backward() # 反向传播 沿计算图得到所有参数的梯度
    optim.step() # 优化

    loss_sum += loss

print(loss_sum / len(test_loader))
```

我们可以进行多轮学习

```python

loss_fn = nn.CrossEntropyLoss()

epochs = 20

for epoch in range(epochs):
    loss_sum = 0
    for data in test_loader: # 按batch_size数量进行遍历

        # 正向推理
        imgs, targets = data
        outputs = mynn(imgs)

        # 计算得到损失函数
        loss = loss_fn(outputs, targets)

        optim.zero_grad() # 将所有参数进行梯度清零
        loss.backward() # 反向传播 沿计算图得到所有参数的梯度
        optim.step() # 优化

        loss_sum += loss

    print('step', epoch, ' = ', loss_sum / len(test_loader))
```

当loss收敛后，就完成了训练



## 模型的修改

 除了自己的模型，其实也能修改别人训练完的模型

以此网络为例：

```python
import torch
from torch import nn

class Mynn(nn.Module):

    def __init__(self):
        super().__init__()
        self.sequential1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(), # 将tensor张成一维张量
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )
        self.sequential2 = nn.Linear(10,10)
        self.classfication = nn.Sequential(
            nn.Linear(10,10),
            nn.Linear(10,10),
            nn.Linear(10,10)
        )


mynn = Mynn()
print(mynn)

```

网络结构为：

```python
Mynn(
  (sequential1): Sequential(
    (0): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (2): Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Flatten(start_dim=1, end_dim=-1)
    (7): Linear(in_features=1024, out_features=64, bias=True)
    (8): Linear(in_features=64, out_features=10, bias=True)
  )
  (sequential2): Linear(in_features=10, out_features=10, bias=True)
  (classfication): Sequential(
    (0): Linear(in_features=10, out_features=10, bias=True)
    (1): Linear(in_features=10, out_features=10, bias=True)
    (2): Linear(in_features=10, out_features=10, bias=True)
  )
)
```



### 添加

```python
mynn.add_module('add_ReLU', nn.ReLU() )
print(mynn)

'''
Mynn(
  (sequential1): Sequential(
    (0): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (2): Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Flatten(start_dim=1, end_dim=-1)
    (7): Linear(in_features=1024, out_features=64, bias=True)
    (8): Linear(in_features=64, out_features=10, bias=True)
  )
  (sequential2): Linear(in_features=10, out_features=10, bias=True)
  (classficatipm): Sequential(
    (0): Linear(in_features=10, out_features=10, bias=True)
    (1): Linear(in_features=10, out_features=10, bias=True)
    (2): Linear(in_features=10, out_features=10, bias=True)
  )
  (add_ReLU): ReLU()
)
'''
```

### 修改

```python
mynn.classfication[1] = nn.ReLU()
print(mynn)

'''
Mynn(
  (sequential1): Sequential(
    (0): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (2): Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Flatten(start_dim=1, end_dim=-1)
    (7): Linear(in_features=1024, out_features=64, bias=True)
    (8): Linear(in_features=64, out_features=10, bias=True)
  )
  (sequential2): Linear(in_features=10, out_features=10, bias=True)
  (classfication): Sequential(
    (0): Linear(in_features=10, out_features=10, bias=True)
    (1): ReLU()
    (2): Linear(in_features=10, out_features=10, bias=True)
  )
  (add_ReLU): ReLU()
)
'''
```



## 模型的保存与读取



### save

```python
import torch


torch.save(model, 'new_model.pth')
model = torch.load('my_model.pth')
```

完整保存了模型的结构与参数，数据量大



### state_dict（官方推荐）

```python
import torch
model = MyModel() # 需要保证与读取的模型是同一个类

# 以字典形式进行 更加方便
torch.save(model.state_dict(), 'model_state_dict1.pth')

state_dict = torch.load('model_state_dict.pth')
model.load_state_dict(state_dict)



```

