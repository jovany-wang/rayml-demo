
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

from torch.utils.data import DataLoader, DistributedSampler

# For Ray
from ray import train
import ray.train.torch
from ray.train import Trainer
from ray.train.torch import TorchConfig

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

    train_loader = DataLoader(train_data, batch_size=32)
    train_loader = train.torch.prepare_data_loader(train_loader)
    return train_loader    

def _prepare_test_data():
    transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5],std=[0.5])])
    test_data = datasets.MNIST(root="/tmp/raytrain_demo/data/",
                            transform = transform,
                            train = False)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=32,
                                            shuffle=True,num_workers=2)
    return test_loader

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
        self.fc1 = nn.Linear(64*7*7,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,10)
#         self.dp = nn.Dropout(p=0.5)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 64 * 7* 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))   
        x = self.fc3(x)  
        return x

def train_func(config):
    train_loader = _prepare_train_data()
    
    cnn_net = CNN()
    cnn_net = train.torch.prepare_model(cnn_net)

    # 3. Define the loss function.
    criterion = nn.CrossEntropyLoss()
    # 4. Define the SGD optimizer.
    sgd_optimizer = optim.SGD(cnn_net.parameters(), lr=0.001, momentum=0.9)

    train_accs = []
    train_loss = []
    test_accs = []
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # net = net.to(device)
    net = cnn_net
    optimizer = sgd_optimizer

    for epoch in range(5):
        running_loss = 0.0
        for i,data in enumerate(train_loader, 0):
            inputs,labels = data
            # Cleanup the data of the last batch
            optimizer.zero_grad()         
            # forward, backward, optimizing
            outputs = net(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d,%5d] loss :%.3f' %
                    (epoch+1,i+1,running_loss/100))
                running_loss = 0.0
            train_loss.append(loss.item())
            
            correct = 0
            total = 0
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item() # Number of the correct predication items.
            train_accs.append(100*correct/total) 
    # # Draw the graph of the trained model
    # train_iters = range(len(train_accs))
    # _draw_train_process('training',train_iters,train_loss,train_accs,'training loss','training acc')
    print('Finished Training')
    return net


def _prepare_data_and_train():
    trainer = Trainer(backend="torch", num_workers=2)
    trainer.start() # set up resources
    trainer.run(train_func)
    trainer.shutdown() # clean up resources
    print("==============OK")

    # trained_net = _train_my_model(cnn_net, train_loader, sgd_optimizer, criterion)
    # path_to_save = "/tmp//raytrain_demo/trainedmodel"
    # torch.save(trained_net.state_dict(), path_to_save)

def _load_model_and_predict():
    test_loader = _prepare_test_data() 
    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    print('GroundTruth: ', ' '.join('%d' % labels[j] for j in range(64)))
    test_net = CNN()
    test_net.load_state_dict(torch.load("/tmp//raytrain_demo/trainedmodel"))
    test_out = test_net(images)

    print(test_out)
    _, predicted = torch.max(test_out, dim=1)
    print('Predicted: ', ' '.join('%d' % predicted[j]
                                for j in range(64)))
                    
if __name__ == "__main__":
    ray.init()
    _prepare_data_and_train()
