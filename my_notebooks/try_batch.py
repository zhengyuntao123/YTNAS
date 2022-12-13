import os, pickle, sys
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import glob
from prettytable import PrettyTable
from tqdm import tqdm
from torch.utils.data import Dataset

import torch
from torch import nn
from torch.nn import Linear,Sequential
from torch.utils.data import DataLoader
import torch.nn.functional as func
import time
from math import isnan,isinf,log,exp
import random

fname=r'../myp/tensors.p'
rname='CIFAR10'

runs_cifar10=[]
f = open(fname,'rb')
while(1):
    try:
        runs_cifar10.append(pickle.load(f))
    except EOFError:
        break
f.close()
print(fname, len(runs_cifar10))

x_data_cifar10=[]
y_data_cifar10=[]
for idx in range(len(runs_cifar10)):
    temp=runs_cifar10[idx]['logmeasures']['synflow']
    for i in range(len(temp)):
        temp[i]=temp[i].item()
    temp=torch.tensor(temp,dtype=torch.float64)
    x_data_cifar10.append(temp)
    y_data_cifar10.append(runs_cifar10[idx]['testacc'])
x_data_cifar10=torch.tensor([item.cpu().detach().numpy() for item in x_data_cifar10],dtype=torch.float64).cuda()
y_data_cifar10=torch.tensor(y_data_cifar10,dtype=torch.float64)
y_data_cifar10=y_data_cifar10/100

trainx_cifar10=[]
trainy_cifar10=[]
for i in range (len(x_data_cifar10)):
    if isinf(torch.sum(x_data_cifar10[i]).item()):
        continue
    trainx_cifar10.append(x_data_cifar10[i])
    trainy_cifar10.append(y_data_cifar10[i])
trainx_cifar10=torch.tensor([item.cpu().detach().numpy() for item in trainx_cifar10],dtype=torch.float64)
trainy_cifar10=torch.tensor(trainy_cifar10,dtype=torch.float64)
trainy_cifar10=torch.reshape(trainy_cifar10,(len(trainy_cifar10),1))
print(len(trainx_cifar10),len(trainy_cifar10))

split_rate=0.2
temp=int(len(trainx_cifar10)*(1-split_rate))
testx_cifar10=trainx_cifar10[temp:]
trainx_cifar10=trainx_cifar10[:temp]
testy_cifar10=trainy_cifar10[temp:]
trainy_cifar10=trainy_cifar10[:temp]
print(len(trainx_cifar10),len(testx_cifar10))

trainx_cifar10=trainx_cifar10.cuda()
trainy_cifar10=trainy_cifar10.cuda()
testx_cifar10=testx_cifar10.cuda()
testy_cifar10=testy_cifar10.cuda()


class MyData(Dataset):
    def __init__(self, trainx, trainy):
        self.trainx = trainx
        self.trainy = trainy

    def __getitem__(self, idx):
        return self.trainx[idx], self.trainy[idx]

    def __len__(self):
        return len(self.trainx)

train_dataset=MyData(trainx_cifar10,trainy_cifar10)
# test_dataset=MyData(testx_cifar10,testy_cifar10)

#这句话很关键，nn默认类型为float32
torch.set_default_dtype(torch.float64)


# 损失函数
class SpearmanLossFunc(nn.Module):
    def __init__(self):
        super(SpearmanLossFunc, self).__init__()

    def forward(self, t1, t2):
        t1 = t1.cpu().detach().numpy()
        t2 = t2.cpu().detach().numpy()
        loss = -abs(stats.spearmanr(t1, t2, nan_policy='omit').correlation)
        loss = torch.tensor(loss, requires_grad=True)
        return loss


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, t1, t2):
        t1 = t1.cpu().detach().numpy()
        t2 = t2.cpu().detach().numpy()
        loss = 0
        #         sample_size=500
        #         sample=random.sample(range(len(trainx)),sample_size)
        t1 = t1.reshape(-1, )
        t2 = t2.reshape(-1, )
        for i in range(len(t1)):
            for j in range(len(t2)):
                if i == j:
                    continue
                loss += log(1 + exp(-np.sign((t1[i] - t1[j]) * (t2[i] - t2[j]))))
        # loss=loss/(sample_size*(sample_size-1))
        loss = torch.tensor(loss, requires_grad=True)
        return loss


class zyt(nn.Module):
    def __init__(self):
        super(zyt, self).__init__()
        self.model1 = Sequential(
            nn.LayerNorm(188),
            Linear(188, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            Linear(128, 64),
            #             nn.BatchNorm1d(64),
            nn.ReLU(),
            Linear(64, 10),
            nn.ReLU(),
            Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model1(x)
        return x


zyt1 = zyt()
zyt1 = zyt1.cuda()

# loss_fn=torch.nn.MSELoss()#SpearmanLossFunc()
loss_fn=MyLoss()
loss_fn=loss_fn.cuda()
#优化器
learning_rate=0.01
optimizer=torch.optim.Adam(zyt1.parameters(),lr=learning_rate)
#训练的轮数
epoch=100
#batch
batch_size=64

train_dataloader=DataLoader(train_dataset,batch_size=batch_size)

maxspear=0
for i in range(epoch):
    epoch_output=[]
    epoch_loss=0
    zyt1.train()
    print("------第{}轮训练开始-------".format(i+1))
#     start_time=time.time()
    total_train_loss=0
    for data in train_dataloader:
        train_data,target=data
        train_data.cuda()
        target.cuda()
        output=zyt1(train_data)
        epoch_output.append(output)
        loss=loss_fn(output,target)
        epoch_loss+=loss
        #优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_output=[j.item() for i in epoch_output for j in i]
    curspear=abs(stats.spearmanr(epoch_output,trainy_cifar10.cpu().detach().numpy(),nan_policy='omit').correlation)
    print(curspear)
    if (curspear>maxspear):
        maxspear=curspear
        maxepoch=i
    print("train_loss:{}".format(epoch_loss.item()))
#测试步骤开始
zyt1.eval() #让网络进入测试状态
#no_grad()保证不影响梯度，不会进行调优
with torch.no_grad():
    test_output=zyt1(testx_cifar10)
    test_loss=loss_fn(test_output,testy_cifar10)
    print("test_loss:{}".format(test_loss.item()))