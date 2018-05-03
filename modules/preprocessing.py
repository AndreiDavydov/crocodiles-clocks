import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread 
import os

def Preprocessing(seed=0):
    '''
    Function reads the data from folders, assigns labels and shuffle whole arrays properly for the following splitting.
    '''
    np.random.seed(0)
    clocklist = [file for file in os.listdir('clocks_crocodiles/clock')]
    croclist = [file for file in os.listdir('clocks_crocodiles/crocodile')]

    clocks_data     = np.array([np.array(imread('clocks_crocodiles/clock/'    +file)/255) for file in clocklist])
    crocodiles_data = np.array([np.array(imread('clocks_crocodiles/crocodile/'+file)/255) for file in croclist ])
    
    X, y = np.concatenate((clocks_data, crocodiles_data),axis=0), np.concatenate((np.zeros(500),np.ones(500)))

    idx = np.arange(1000)
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]
    return X, y

def get_crocs():
    '''
    Reads and returns only Crocodile set.
    '''
    croclist = [file for file in os.listdir('clocks_crocodiles/crocodile')]
    crocodiles_data = np.array([np.array(imread('clocks_crocodiles/crocodile/'+file)/255) for file in croclist ])
    return crocodiles_data

def get_clocks():
    '''
    Reads and returns only Clock set.
    '''
    clocklist = [file for file in os.listdir('clocks_crocodiles/clock')]
    clocks_data     = np.array([np.array(imread('clocks_crocodiles/clock/'    +file)/255) for file in clocklist])
    return clocks_data

def check_preprocessing(X, y, seed=0):
    '''
    Takes 5 images from X chosed randomly and shows them with labels.
    It is addressed to prove correct labelling on the preprocessing part.
    '''
    np.random.seed(seed)
    fig, ax = plt.subplots(1,5,figsize=(20,10))
    for i in range(5):
        idx = np.random.randint(0,1000)
        ax[i].imshow(X[idx])
        ax[i].set_title('crocodile' if int(y[idx])==1 else 'clock')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.show()
        
from sklearn.model_selection import train_test_split

def swap(arr):
    '''
    Takes the 4-dimensional array (N, (image in numpy shape), 3) as input and swaps its dimensions to make it properly prepared for implementation into Torch net.
    Opposite to "swap_back" function.
    '''
    arr = np.swapaxes(arr, 2,3)
    arr = np.swapaxes(arr, 1,2)
    return arr

def swap_back(arr):
    '''
    Takes the 4-dimensional array (N,3,(image in numpy shape)) as input and swaps its dimensions to make it properly prepared for visualization with "imshow" function in matplotlib.
    Opposite to "swap" function.
    '''
    arr = np.swapaxes(arr, 1,2)
    arr = np.swapaxes(arr, 2,3)
    return arr
   


























        

