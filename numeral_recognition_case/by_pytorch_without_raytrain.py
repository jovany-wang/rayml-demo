
import numpy as np

import logging as logger

logger.basicConfig()
logger.root.setLevel(logger.INFO)
logger.basicConfig(level=logger.INFO)


import torch 
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
import os
from torchvision import datasets, transforms,utils
import torch.nn.functional as F
import torch.optim as optim

def _draw_train_process(title,iters,costs,accs,label_cost,lable_acc):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("acc(\%)", fontsize=20)
    plt.plot(iters, costs,color='red',label=label_cost) 
    plt.plot(iters, accs,color='green',label=lable_acc) 
    plt.legend()
    plt.grid()
    plt.show()


def _prepare_train_data():
    transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5],std=[0.5])])
    train_data = datasets.MNIST(root = "/tmp/raytrain_demo/data/",
                                transform=transform,
                                train = True,
                                download = True)

    train_loader = torch.utils.data.DataLoader(train_data,batch_size=64,
                                            shuffle=True,num_workers=2)
    return train_loader    

def _prepare_test_data():
    transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5],std=[0.5])])
    test_data = datasets.MNIST(root="/tmp/raytrain_demo/data/",
                            transform = transform,
                            train = False)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=64,
                                            shuffle=True,num_workers=2)
    return test_loader

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
        self.fc1 = nn.Linear(64*7*7,1024) #两个池化，所以是7*7而不是14*14
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,10)
#         self.dp = nn.Dropout(p=0.5)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 64 * 7* 7)#将数据平整为一维的 
        x = F.relu(self.fc1(x))
#         x = self.fc3(x)
#         self.dp(x)
        x = F.relu(self.fc2(x))   
        x = self.fc3(x)  
#         x = F.log_softmax(x,dim=1) NLLLoss()才需要，交叉熵不需要
        return x

def _train_my_model(net, train_loader, optimizer, criterion):
    train_accs = []
    train_loss = []
    test_accs = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    for epoch in range(5):
        running_loss = 0.0
        for i,data in enumerate(train_loader, 0):#0是下标起始位置默认为0
            # data 的格式[[inputs, labels]]       
    #         inputs,labels = data
            inputs,labels = data[0].to(device), data[1].to(device)
            #初始为0，清除上个batch的梯度信息
            optimizer.zero_grad()         
            #前向 + 后向 + 优化     
            outputs = net(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            # loss 的输出，每个一百个batch输出，平均的loss
            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d,%5d] loss :%.3f' %
                    (epoch+1,i+1,running_loss/100))
                running_loss = 0.0
            train_loss.append(loss.item())
            
            # 训练曲线的绘制 一个batch中的准确率
            correct = 0
            total = 0
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)# labels 的长度
            correct = (predicted == labels).sum().item() # 预测正确的数目
            train_accs.append(100*correct/total) 
    # # 绘制出模型效果图
    # train_iters = range(len(train_accs))
    # _draw_train_process('training',train_iters,train_loss,train_accs,'training loss','training acc')
    print('Finished Training')
    return net


def _prepare_data_and_train():
    # 1. Pprepare data.
    train_loader = _prepare_train_data()
    # 2. Define a CNN model.
    cnn_net = CNN()
    # 3. Define the loss function.
    criterion = nn.CrossEntropyLoss()
    # 4. Define the SGD optimizer.
    sgd_optimizer = optim.SGD(cnn_net.parameters(), lr=0.001, momentum=0.9)
    # 也可以选择Adam优化方法
    # optimizer = torch.optim.Adam(net.parameters(),lr=1e-2)
    
    # 5. Train my CNN model.
    trained_net = _train_my_model(cnn_net, train_loader, sgd_optimizer, criterion)
    path_to_save = "/tmp//raytrain_demo/trainedmodel"
    torch.save(trained_net.state_dict(), path_to_save)

def _load_model_and_predict():
    test_loader = _prepare_test_data() 
    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    print('GroundTruth: ', ' '.join('%d' % labels[j] for j in range(64)))
    test_net = CNN()
    test_net.load_state_dict(torch.load("/tmp//raytrain_demo/trainedmodel"))
    test_out = test_net(images)
    # 输出的是每一类的对应概率，所以需要选择max来确定最终输出的类别 dim=1 表示选择行的最大索引
    print(test_out)
    _, predicted = torch.max(test_out, dim=1)
    print('Predicted: ', ' '.join('%d' % predicted[j]
                                for j in range(64)))
                    
if __name__ == "__main__":
    _prepare_data_and_train()
