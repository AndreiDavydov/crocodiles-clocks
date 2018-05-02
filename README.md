# Clocks and Crocodiles

## My name is Andrei Davydov and here is my project report for Skoltech summer internship in Samsung. The project is devoted to the research based on the dataset with images of clocks and crocodiles. 

The project consists of three parts: 
  - binary **classification** of images (1), 
  - **extraction** of the images that are debateable on the issue, which label must be assigned to (2),
  - **generation** of new images that would hardly correspond to one of two classes (3).
  
Let us talk about each part and my achievements on them.

### 1. Classification

#### 1.0. Preprocessing

First of all, a proper preprocesssing must be achieved. Each image from folders must have its true label and all images must be stacked together in the whole array. Then this array must be splitted in the Train, Validation and Test sets for the following classification. All calculations are provided in "preprocessing.py" module, the check result (whether it works properly or not) is provided below:

<p align="center">
  <img width="1000px" src="images4report/check.png">
</p>

#### 1.1. Binary classification with CNN

The core idea was to implement a convolutional neural network, mused by Alexnet and others, as simple as possible due to quite small dataset of quite small images (32x32 size).It consists
