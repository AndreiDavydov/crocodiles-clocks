import numpy as np
import matplotlib.pyplot as plt
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from IPython.display import clear_output

from preprocessing import *
from model_classification import *
    
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        
        self.encoder = nn.Sequential()
        self.encoder.add_module('dense1', nn.Linear(32*32, 1024))
        self.encoder.add_module('dense1_relu', nn.ReLU())
        self.encoder.add_module('dense2', nn.Linear(1024, 1024))
        self.encoder.add_module('dense2_relu', nn.ReLU())

        nn.init.xavier_uniform(self.encoder.dense1.weight)
        nn.init.xavier_uniform(self.encoder.dense2.weight)
        self.encoder.dense1.bias.data.zero_()
        self.encoder.dense2.bias.data.zero_()

        self.decoder = nn.Sequential()
        self.decoder.add_module('dense1', nn.Linear(1024, 1024))
        self.decoder.add_module('dense1_relu', nn.ReLU())
        self.decoder.add_module('dense2', nn.Linear(1024, 32*32))
        self.decoder.add_module('dense2_relu', nn.Sigmoid())

        nn.init.xavier_uniform(self.decoder.dense1.weight)
        nn.init.xavier_uniform(self.decoder.dense2.weight)
        self.decoder.dense1.bias.data.zero_()
        self.decoder.dense2.bias.data.zero_()
        
    def forward(self, x):
        
        latent_code    = self.encoder(x)
        reconstruction = self.decoder(latent_code)
        
        return reconstruction, latent_code
    
def training_process_ae(model, X, num_epochs=20):
    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters())
    
    train_loss = []
    initial_img = Variable(torch.FloatTensor(X[0])).cuda()
    initial_img = initial_img.view(1,3,32*32)
    model.train(True)
    start = time()
    X = Variable(torch.FloatTensor(X)).cuda()
    X = X.view(X.size(0),3,32*32)
    for epoch in range(num_epochs):
        X_pred = model(X)[0]
        loss = criterion(X_pred, X)
        loss.backward()
        opt.step()
        opt.zero_grad()
        
        train_loss.append(loss.data.cpu().numpy()[0])       
        
        time_ = time()-start
        clear_output(True)
        fig, ax = plt.subplots(1, 3, figsize=(15,5))
        ax[0].plot(train_loss, 'b-')
        ax[0].set_title('epoch {} took {:.2f}s. Loss = {:.5f}'.format(epoch, time_, loss.data.cpu().numpy()[0]))
        before, after = initial_img.data.cpu().numpy().reshape(1,3,32,32), \
                        model(initial_img)[0].data.cpu().numpy().reshape(1,3,32,32)
        before, after = swap_back(before)[0], swap_back(after)[0]
        ax[1].imshow(before)
        ax[2].imshow(after)
        plt.show()