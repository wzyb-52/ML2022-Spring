# Notes of ML2022

[TOC]

## Introduction

This file marks down all the questions raised in implementing the homeworks.

## Basic concepts

+ Different types of functions:

  + Regression: The function outputs a scalar.
  + Classification: Gives options(classes), the function outputs the correct one.
  + Structured learning: The function creates something with structure(images, documents).

+ Hyperparameters: The parameters set by you.

+ Neuron: Activation functions such as sigmoid and **ReLU**.

+ Symbols in a model:

  + **Feature**, x: the number of the features is the dimension of feature vector.
  + **Parameters**, θ: All unknown parameters of the model.
  + **Loss**, L(θ): A function of parameters, representing how good a set of values is.
  + **Label**, y-head: The true value.
  + **Gradient**, g: ∇L(θ^2^).

+ Actual optimization in training:

  <img src="/home/carolt/SelfEducating/Artificial_Intelligence/ML2022-Spring/Notes/images/image-20230204003128303.png" alt="image-20230204003128303" style="zoom:80%;" />

  + Batch: The computation unit to update parameters.
  + Shuffle: Shuffle after each epoch to make batches different among epochs.
  + Update: The process of updating parameters according to one batch of data.
  + Epoch: The process of updating parameters according to whole data.

+ Criterion on model:

  + Overfitting: Better on training data, worse on unseen data.

## Lecture 1: Introduction of Deep Learning

### Question 1: Device ---- GPU or CPU?

Tensors & modules will be computed with CPU by default.

+ Use `.to()` to move tensors to appropriate devices.

  ```python
  x = x.to("cpu")
  x = x.to("gpu")
  ```

+ Check if your computer has NVIDIA GPU:

  ```python
  torch.cuda.is_available()

So, which device to choose to finish the deep learning tour?

+ RTX 2060, GTX 1660 / 1060 / 1050: personal independent GPU. Available for local debug and test. Price ranges from 1700 yuan to 459 yuan(including second hand ones). Satisfying for some potential game demands in future.
+ Cloud GPUs: using or renting resources from labs or companies. Dedicated for experiment or work demands. Free or cheap in a short term, while long-term renting cost is a big budget. Relatively more inconvenient when having to share resources with others.

### Question 2: Training & testing neural network

There are 4 steps to train and test a neural network.

1. Load data.

   <img src="/home/carolt/SelfEducating/Artificial_Intelligence/ML2022-Spring/Notes/images/image-20230202164940376.png" alt="image-20230202164940376" style="zoom:80%;" />

   Dataset stores data samples and expected values, while dataloader groups data in batches and enables multiprocessing.

   > Why use **batch** and **shuffle** in the process of training?
   >
   > 

2. Define neural network.

   <img src="/home/carolt/SelfEducating/Artificial_Intelligence/ML2022-Spring/Notes/images/image-20230202195537568.png" alt="image-20230202195537568" style="zoom:80%;" />

   > Why input tensor can be any shape but last dimension must be 32? What do the dimensions of input tensor and `nn.Linear()` represent?
   >
   > <img src="/home/carolt/SelfEducating/Artificial_Intelligence/ML2022-Spring/Notes/images/image-20230202195751135.png" alt="image-20230202195751135" style="zoom:80%;" />
   >
   > 32 is the number of input features. 64 is the number of output features.

3. Loss function.

4. Optimization algorithm.

   Why we need optimization algorithms such as [Adaptive Learning Rate](https://www.youtube.com/watch?v=HYUXEeh3kwY) and SGD(Stochastic Gradient Descent), and how do they work?

5. Entire procedure.

