import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread 
import os

# First of all, we should preprocess data in the following way:
# 1. read images and assign each point its label, organize X and y, then shuffle.
# 2. split the data into Train, Validation and Test parts
# 3. prepare the data for Torch handling - permute axis, etc.

def Preprocessing():
    clocklist = [file for file in os.listdir('clocks_crocodiles/clock')]
    croclist = [file for file in os.listdir('clocks_crocodiles/crocodile')]

    # let's say, '0' corresponds to 'clock', '1' - to 'crocodile' 
    clocks_data     = np.array([np.array(imread('clocks_crocodiles/clock/'    +file)/255) for file in clocklist])
    crocodiles_data = np.array([np.array(imread('clocks_crocodiles/crocodile/'+file)/255) for file in croclist ])
    
    X, y = np.concatenate((clocks_data, crocodiles_data),axis=0), np.concatenate((np.zeros(500),np.ones(500)))

    idx = np.arange(1000)
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]
    return X, y

def get_crocs():
    croclist = [file for file in os.listdir('clocks_crocodiles/crocodile')]
    crocodiles_data = np.array([np.array(imread('clocks_crocodiles/crocodile/'+file)/255) for file in croclist ])
    return crocodiles_data

def get_clocks():
    clocklist = [file for file in os.listdir('clocks_crocodiles/clock')]
    clocks_data     = np.array([np.array(imread('clocks_crocodiles/clock/'    +file)/255) for file in clocklist])
    return clocks_data

def check_preprocessing(X, y, seed=0):
    np.random.seed(seed)
    fig, ax = plt.subplots(1,5,figsize=(20,10))
    for i in range(5):
        idx = np.random.randint(0,1000)
        ax[i].imshow(X[idx])
        ax[i].set_title('crocodile' if int(y[idx])==1 else 'clock')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        
from sklearn.model_selection import train_test_split

def swap(arr):
    arr = np.swapaxes(arr, 2,3)
    arr = np.swapaxes(arr, 1,2)
    return arr

def swap_back(arr):
    arr = np.swapaxes(arr, 1,2)
    arr = np.swapaxes(arr, 2,3)
    return arr
   


























        

