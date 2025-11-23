import time

from torchvision.datasets import FashionMNIST
import numpy as np
from torchvision import transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
from model import AlexNet
import torch
import torch.nn as nn
import copy
import pandas as pd

def train_val_data_process():                                        
    train_data = FashionMNIST('./data',
                              train=True,
                              download=True,
                              transform=transforms.Compose([transforms.Resize(size=227),
                                                            transforms.ToTensor()]))

    train_data, val_data = data.random_split(train_data, [int(len(train_data) * 0.8),int(len(train_data) * 0.2)])

    train_loader = data.DataLoader(train_data,
                                   batch_size=64,
                                   shuffle=True,
                                   num_workers=8)
    val_loader = data.DataLoader(val_data,
                                   batch_size=64,
                                   shuffle=True,
                                   num_workers=8)
    return train_loader, val_loader

def train_model_process(model,train_loader,val_loader,num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #自动调整学习率

    criterion = nn.CrossEntropyLoss()   #交叉熵损失函数   用于分类
    #将模型放到设备中
    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())

    #初始化参数
    best_acc = 0.0
    #训练集损失函数列表
    train_loss_all = []
    #训练集损失列表
    val_loss_all = []
    #训练集准确度列表
    train_acc_all = []
    #训练集准确度列表
    val_acc_all = []

    since= time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print("-" * 10)

        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0

        train_num = 0
        val_num = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            model.train()
            output = model(inputs)

            pre_lab = torch.argmax(output, dim=1)

            loss = criterion(output, targets)

            #将梯度初始化为0
            optimizer.zero_grad()
            #反向传播计算
            loss.backward()

            #利用梯度下降法
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_acc += torch.sum(pre_lab == targets)

            train_num += inputs.size(0)


        for step, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            #设置为评估模式
            model.eval()
            #前向传播的过程
            output = model(inputs)

            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, targets)
            val_loss += loss.item() * inputs.size(0)
            val_acc += torch.sum(pre_lab == targets)
            val_num += inputs.size(0)

        #计算并保存每一次迭代的loss值和准确率
        train_loss_all.append(train_loss / train_num)
        val_loss_all.append(val_loss / val_num)
        train_acc_all.append(train_acc.double().item() / train_num)
        val_acc_all.append(val_acc.double().item() / val_num)

        print('{} Train Loss:{:.4f} Train Acc:{:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1] ))
        print('{} Val Loss:{:.4f} Val Acc:{:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1] ))

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1] #更新检测度
            best_model_wts = copy.deepcopy(model.state_dict())

        time_used = time.time() - since
        print("训练和验证耗时的时间{:.0f}m {:.0f}s".format(time_used // 60, time_used % 60))


    #save the param

    torch.save(best_model_wts, 'F:/DL/AleNet/model.pth') #权重文件

    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                        "train_loss": train_loss_all,
                                        "train_acc": train_acc_all,
                                        "val_loss": val_loss_all,
                                        "val_acc": val_acc_all})


    return train_process


def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process.epoch,train_process.train_loss,'ro-',label='train loss')
    plt.plot(train_process.epoch,train_process.val_loss,'bs-',label='val loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')


    plt.subplot(1,2,2)
    plt.plot(train_process.epoch,train_process.train_acc,'ro-',label='train acc')
    plt.plot(train_process.epoch,train_process.val_acc,'bs-',label='val acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


if __name__ == '__main__':
        #加载模型

    AlexNet = AlexNet()
    train_loader, val_loader = train_val_data_process()
    train_process = train_model_process(AlexNet,train_loader,val_loader,10)
    matplot_acc_loss(train_process)









