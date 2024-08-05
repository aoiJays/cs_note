import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import pandas as pd
import os
from PIL import Image

# class base_resnet(nn.Module):
#     def __init__(self):
#         super(base_resnet, self).__init__()
#         self.model = models.resnet50(pretrained=True)  # 更改为ResNet18

#         # 修改最后一层
#         self.model.fc = nn.Sequential(
#             nn.Linear(512, 1024, bias=True),  # ResNet18的最后一层输入特征数量为512
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Linear(1024, 176, bias=True)
#         )
        
#         # 将其他层的参数设置为不需要更新
#         for param in self.model.parameters():
#             param.requires_grad = False

#         # 只训练最后两层
#         for param in self.model.layer4.parameters():
#             param.requires_grad = True
#         for param in self.model.fc.parameters():
#             param.requires_grad = True

#     def forward(self, x):
#         x = self.model(x) 
#         return x

class base_resnet(nn.Module):
    def __init__(self):
        super(base_resnet, self).__init__()
        self.model = models.resnet50(pretrained=True)
        # 修改最后一层
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 176, bias=True)
        ) 
        
        # 将其他层的参数设置为不需要更新
        for param in self.model.parameters():
            param.requires_grad = False

        # 只训练最后两层
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        for param in self.model.fc.parameters():
            param.requires_grad = True
 
    def forward(self, x):
        x = self.model(x) 
        return x
    
    
# 数据路径
train_csv = '/kaggle/input/classify-leaves/train.csv'
test_csv = '/kaggle/input/classify-leaves/test.csv'  # 假设你有一个测试集的CSV文件
images_dir = '/kaggle/input/classify-leaves/'

train_data = pd.read_csv(train_csv)
test_data = pd.read_csv(test_csv) 

categories = pd.unique(train_data['label']).tolist()
categories.sort()

train_data = pd.concat([train_data.drop('label', axis=1), pd.get_dummies(train_data['label'], prefix='label').astype('float64')], axis=1)
train_data.head()

class LeafDataset(Dataset):
    def __init__(self, data_frame, root_dir):
        self.data_frame = data_frame
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: x / 255.0) 
        ])
        
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])  # image path
        image = Image.open(img_name)
        label = self.data_frame.iloc[idx, 1:].tolist()
        
        image = self.transform(image)
        label = torch.tensor(label,dtype=torch.float32)
        
        return image, label
    
    
train_dataset = LeafDataset(train_data, images_dir)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
print(train_size, val_size)

train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
batch_size = 128

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,num_workers=4,pin_memory=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True,num_workers=4,pin_memory=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
model = base_resnet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

def training(dataloader, model, loss_fn, optimizer):
    
    model.train()

    loss, correct= 0.0, 0.0
    n = 0

    for X,y in dataloader:
    
        X,y = X.to(device), y.to(device)
        output = model(X)

        cur_loss = loss_fn(output, y)
        _, pred = torch.max(output, axis = 1)
        _, ypred = torch.max(y, axis = 1)
            
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        loss += cur_loss * len(y)
        correct += torch.sum(ypred==pred).item()
        n += len(y)

    return loss / n, correct / n

def val(dataloader, model, loss_fn):
    
    model.eval()

    with torch.no_grad():

        loss, correct= 0.0, 0.0
        n = 0 

        for X,y in dataloader:

            X,y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            output = model(X)
            _, pred = torch.max(output, axis = 1)
            _, ypred = torch.max(y, axis = 1)
            
            cur_loss = loss_fn(output, y)
            correct += torch.sum(ypred==pred).item()
            loss += cur_loss * len(y)
            n += len(y)
            

            
    return loss/n, correct/n


epochs = 5
max_acc = 0

params = model.state_dict()

for epoch in range(epochs):

    loss, acc = training(train_loader, model, loss_fn, optimizer)
    print(f'epoch {epoch + 1}/{epochs}: loss = {loss} acc = {acc}')
    loss, acc = val(val_loader, model, loss_fn)
    print(f'Validation: loss = {loss} acc = {acc}\n')


    if acc > max_acc:
        max_acc = acc
        params = model.state_dict()
        print('saved local best')
        
print(max_acc)

model.load_state_dict(params)

class LeafTestDataset(Dataset):
    def __init__(self, data_frame, root_dir):
        self.data_frame = data_frame
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: x / 255.0) 
        ])
        
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])  # image path
        image = Image.open(img_name)
        
        image = self.transform(image)
        
        return image
test_dataset = LeafTestDataset(test_data, images_dir)



res = []

with torch.no_grad():
    n = len(test_dataset)
    for i in range(n):
        x = test_dataset[i].reshape(1,3,224,224).cuda()
        output = model(x)
        _,pred = torch.max(output, axis = 1)
        res.append(categories[pred.item()])
        # if len(res) > 10 : break
        
df = test_data['image']
res_df = pd.DataFrame({'result': res})
combined_df = pd.concat([test_data, res_df], axis=1)  
combined_df[:20]

combined_df.columns = ['image','label']

combined_df.to_csv('/kaggle/working/output.csv', encoding='utf-8', index=False)