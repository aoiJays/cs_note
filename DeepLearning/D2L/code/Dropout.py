import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import time

# 定义数据预处理
trans = transforms.Compose([
    transforms.ToTensor()
])

# 加载数据集
train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True,
                                                  transform=trans, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False,
                                                 transform=trans, download=True)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self):
        
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 28*28)
        self.fc2 = nn.Linear(28*28, 10)
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.3)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        
        x = self.fc1(x)
        x = self.ReLU(x)
        x = self.dropout(x) # 隐藏层的输出

        x = self.fc2(x)
        x = self.ReLU(x)
        return x
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

print(f'Traing Device {device}')

model = MLP().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.03)

# 训练模型

start = time.time()

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 测试模型
    model.eval()
    with torch.no_grad():  

        correct = 0
        total = 0

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        print(f'Accuracy of the model on the test images: {100 * correct / total} %')
print(f'Traing Time = {time.time() - start}')