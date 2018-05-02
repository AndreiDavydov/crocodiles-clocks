# Clocks and Crocodiles

## My name is Andrei Davydov and here is my project report for Skoltech summer internship in Samsung. 

## The project is devoted to the research based on the dataset with images of clocks and crocodiles. 

The project consists of three parts: 
  - binary **classification** of images (1), 
  - **extraction** of the images that are debateable on the issue, which label must be assigned to (2),
  - **generation** of new images that would hardly correspond to one of two classes (3).
  
Let us talk about each part and my achievements on them. 

*All implementation is written on the Python (3.6) using ordinary libraries Numpy (1.14.0), Matplotlib (2.1.2), Scikit (0.19.1
) and also the library specifically developed for comfortable neural network implementation - pyTorch (0.3.0.). All NN-based calculations were performed with CUDA (9.0) driver on the GPU Nvidia GeForce GTX 950m for faster evaluation.*

### 1. Classification

#### 1.0. Preprocessing

First of all, a proper preprocesssing must be achieved. Each image from folders must have its true label (let's say, '0' corresponds to 'clock', '1' - to 'crocodile') and all images must be stacked together in the whole array. Then this array must be splitted in the Train, Validation and Test sets for the following classification. All calculations are provided in "preprocessing.py" module, the checking result (whether it works properly or not) is provided below:

<p align="center">
  <img width="1000px" src="images4report/check.png">
</p>

#### 1.1. Binary classification with CNN

The core idea was to implement a convolutional neural network, mused by Alexnet and others, as simple as possible due to a relatively small dataset (500 for each class) of relatively small images (32x32 pixels).It consists of three blocks of **Conv2d+MaxPool2d+ReLU** (such composition was shown prominent results in image classification tasks, especially proper feature extraction, in the **AlexNet** for instance). Then these blocks are flattened and followed by two **dense** layers with nonlinearities. The last layer is **sigmoid** function, which eventually learns to give the probabilities of each image be labelled to '0'th or '1'th class. A **Binary Cross-Entropy** loss - a pretty common criterion for the classification optimization procedure was used with the **Adam** optimizer. All the code for model implementation and learning is provided in the "model_classification.py" module. A visualization of the training procedure is provided below:

<p align="center">
  <img width="1000px" src="images4report/loss_acc.png">
</p>

The model was saved in "model_state" file and can be reloaded from it by simple pyTorch functions (".load_state_dict(torch.load('model_state')...").

Afterwards, accuracies on each dataset were calculated:
- train set - **91.8 %**
- validation set - **84.4 %**
- test set - **87.2 %**

it should be noted that these results were got without any tricks with data augmentation or CNN structure development. Certainly, results would be much better if more images were in the train set or some efficient layers (as BatchNorm or Dropout) were used. Anyway, even such straightforward approach with so simple CNN structure have shown pretty nice results.

### 2. Extraction of disputable images
