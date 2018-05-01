import numpy as np
import matplotlib.pyplot as plt
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

def compute_loss(net_, X, y):
    '''Computes BCE loss for given vectors X and their labels y'''
    X = Variable(torch.FloatTensor(X)).cuda()
    y = Variable(torch.FloatTensor(y)).cuda()
    y_pred = net_(X)
    return F.binary_cross_entropy(y_pred, y) 

def compute_acc(net_, X, y):
    """Computes the accuracy for multiple binary predictions"""
    X = Variable(torch.FloatTensor(X)).cuda()
    y_pred = net_(X).data.cpu().numpy()
    acc = accuracy_score(y, np.rint(y_pred))
    return acc
        
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.mp1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(True)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.mp2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU(True)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.mp3 = nn.MaxPool2d(2)
        self.relu3 = nn.ReLU(True)        

        self.flatten = Flatten()
        self.lin1 = nn.Linear(256, 256)
        self.lin_relu = nn.ReLU(True)
        self.lin2 = nn.Linear(256, 1)        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        # conv+maxpool+relu layers
        x = self.relu1(self.mp1(self.conv1(x)))
        x = self.relu2(self.mp2(self.conv2(x)))  
        x = self.relu3(self.mp3(self.conv3(x)))           
        
        # fully-connected layers
        x = self.lin1(self.flatten(x))
        x = self.lin2(self.lin_relu(x))
        
        x = self.sigmoid(x)
        return x.view(x.size(0))
    
    def get_latent(self,x):
        x = self.relu1(self.mp1(self.conv1(x)))
        x = self.relu2(self.mp2(self.conv2(x)))  
        x = self.relu3(self.mp3(self.conv3(x)))          
        
        x = self.lin1(self.flatten(x))
        return x

from IPython.display import clear_output
from sklearn.metrics import accuracy_score
    
def training_process(net_, X_train, y_train, X_val, y_val, n_epochs=20):
    opt = optim.Adam(net_.parameters(), lr=0.001)
    loss_history = []
    train_accs = []
    val_accs = []
    net_.train(True)
    for epoch in range(n_epochs):
        
        loss = compute_loss(net_, X_train, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_history.append(loss.data)
        train_accs.append(100*compute_acc(net_, X_train, y_train))
        val_accs.append(100*compute_acc(net_, X_val, y_val))
        
        if (epoch+1) % 10 == 0:      
            net_.train(False)
            train_acc = compute_acc(net_, X_train, y_train)
            val_acc = compute_acc(net_, X_val, y_val)
            clear_output(True)
            fig, ax = plt.subplots(1, 2, figsize=(20,5))
            ax[0].plot(loss_history, '-bo')
            ax[0].set_title('loss vs. epochs')
            ax[0].set_xlabel('# of epochs')
            ax[0].set_ylabel('loss value')
            ax[1].plot(train_accs, '-bo', label='train')
            ax[1].plot(val_accs, '-ro', label='validation')
            ax[1].set_title('train vs. validation accuracy, %')
            ax[1].set_xlabel('# of epochs')
            ax[1].set_ylabel('accuracy, %')
            ax[1].legend(loc='upper center', fontsize='x-large')
            plt.show()
            net_.train(True)
    
    return loss_history     