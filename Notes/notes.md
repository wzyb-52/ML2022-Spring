# Notes of ML2022

[TOC]

## Introduction

This file marks down all the questions raised in implementing the homeworks.

## Basic concepts

+ Different types of functions:

  + Regression: The function outputs a scalar.
  + Classification: Gives options(classes), the function outputs the correct one.
  + Structured learning: The function creates something with structure(images, documents).

+ Symbols in a model:

  + **Feature**, x: Input. The number of the features is the dimension of feature vector.
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
  
+ Others:

  + Neuron: Activation functions such as sigmoid and **ReLU**.
  
  + Hyperparameters: The parameters set by you.
  
  + Saddle points: Non-local extreme points of stationary points.
  
    > Actually, **most critical points are saddle points** instead of local minima, since more dimensions the data have, the harder there is a local minima.

## Lecture 1: Introduction of Deep Learning & Training Tips

### Question 1: Why we want "Deep" network, not "Fat" network?

Here are the experimental data:

> <img src="/home/carolt/SelfEducating/Artificial_Intelligence/ML2022-Spring/Notes/images/image-20230226164746638.png" alt="image-20230226164746638" style="zoom:80%;" />
>
> With the same size of model, "Deep" model is more effective than "Fat" model.

So, why "Deep" network is more effective than "Fat" network with same parameters?

This is because a carefully designed deep network has much more complex and regular structure.

> <img src="/home/carolt/SelfEducating/Artificial_Intelligence/ML2022-Spring/Notes/images/image-20230226165412199.png" alt="image-20230226165412199" style="zoom:80%;" />
>
> For example, to represent 8 straight pieces of lines, deep network could use only 6 ReLU neurons while fat network has to use at least 8 neurons. And if deep network has K layers, which layer has 2 neurons, it is able to represent 2^k pieces.

### Question 2: Device ---- GPU or CPU?

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

### Question 3: Training & testing neural network

There are 4 steps to train and test a neural network.

1. Load data.

   <img src="/home/carolt/SelfEducating/Artificial_Intelligence/ML2022-Spring/Notes/images/image-20230202164940376.png" alt="image-20230202164940376" style="zoom:80%;" />

   Dataset stores data samples and expected values, while dataloader groups data in batches and enables multiprocessing.

   > How to effectively set batch size?
   >
   > + Bigger batch size better exploits the parallel computation capacity of GPU and obtains much better performance on **training speed**, as long as the size of batches are not beyond the limit of GPU. 
   >
   > + However, smaller batch size means more noise in the process of training, bringing greater performance on **both training and testing accuracy**.
   >
   >   + Why smaller batch is better on training data?
   >
   >     > <img src="/home/carolt/SelfEducating/Artificial_Intelligence/ML2022-Spring/Notes/images/image-20230208111448720.png" alt="image-20230208111448720" style="zoom:80%;" />
   >     >
   >     > The horizontal axis representing parameters(θ).
   >     >
   >     > The distributions of batches' data are different from each other. So if one batch stuck on a **saddle point** in the process of updating, the next batches may still be trainable.
   >
   >   + Why smaller batch is better on testing data?
   >
   >     > <img src="/home/carolt/SelfEducating/Artificial_Intelligence/ML2022-Spring/Notes/images/image-20230208112508471.png" alt="image-20230208112508471" style="zoom:80%;" />
   >     >
   >     > One argument is that the distribution of testing data is slightly different from that of training data, so minima of training loss may not exactly be the minima of testing loss. And as far as we can see from the above diagram, **flat minima** has more stable performance when this difference exists. And someone believes that smaller batches prefer to bring us to the flat minima.

2. Define neural network.

   <img src="/home/carolt/SelfEducating/Artificial_Intelligence/ML2022-Spring/Notes/images/image-20230202195537568.png" alt="image-20230202195537568" style="zoom:80%;" />

   > Why input tensor can be any shape but last dimension must be 32? What do the dimensions of input tensor and `nn.Linear()` represent?
   >
   > <img src="/home/carolt/SelfEducating/Artificial_Intelligence/ML2022-Spring/Notes/images/image-20230202195751135.png" alt="image-20230202195751135" style="zoom:80%;" />
   >
   > 32 is the number of input features. 64 is the number of output features.

3. Loss function.

   In classification problem, compared to MSE, **cross-entropy** is widely used as the loss function. And **in PyTorch**, cross-entropy is combined with **softmax** as a set of functions, so we hardly see softmax process in the programs.

   > <img src="/home/carolt/SelfEducating/Artificial_Intelligence/ML2022-Spring/Notes/images/image-20230226153905448.png" alt="image-20230226153905448" style="zoom:80%;" />

   We can see that the error surface of cross-entropy has a much wider range, which means the training process will be faster. For example, the gradient of left-top position of MSE is very small, causing the training **stuck in the beginning**.

4. Optimization algorithm.

   Remember, training stuck do not equal to small gradient. So how to determine whether the parameters are around a critical point?

   > There are some situations that loss doesn't change, while gradient is still vibrating:
   >
   > <img src="/home/carolt/SelfEducating/Artificial_Intelligence/ML2022-Spring/Notes/images/image-20230218145245730.png" alt="image-20230218145245730" style="zoom:80%;" />

   So why we need optimization algorithms such as [Adaptive Learning Rate](https://www.youtube.com/watch?v=HYUXEeh3kwY) and SGD(Stochastic Gradient Descent), and how do they work?

   > As far as we can see from the above figure, the loss always can't keep declining until the parameters reach critical points. So some optimization algorithms such as Adaptive Learning Rate can help us make it.
   >
   > <img src="/home/carolt/SelfEducating/Artificial_Intelligence/ML2022-Spring/Notes/images/image-20230218150826410.png" alt="image-20230218150826410" style="zoom:80%;" />
   >
   > Above picture shows the optimized gradient algorithm, and usually there are three approaches.

5. Entire procedure.

   **General guide** and schemes to accomplish simple tasks:

   > <img src="/home/carolt/SelfEducating/Artificial_Intelligence/ML2022-Spring/Notes/images/image-20230217133546670.png" alt="image-20230217133546670" style="zoom:80%;" />
   >
   > Some questions are as following:
   >
   > + First inspect the loss on training data. If it is too large, we must take some measures to adjust the model.
   >
   >   Following are the situations which may cause large loss on training data:
   >
   >   + Model bias:
   >
   >     For example, if the model is too simple, it can not be flexible enough to have a function making loss small.
   >
   >     The most commonly used improvement strategies are using more features and deeper learning, such as more neurons and layers.
   >
   >   + Improper optimization:
   >
   >     Sometimes complex models perform worse than simple ones **both on training set and testing set**. This situation means improper optimization measures are taken, instead of overfitting.
   >
   >     So how do we determine whether the model is complex enough? Usually we train models in the order from simple to complex. When a more complex model has a large loss on training data, it is the right time to adjust the optimization strategy.
   >
   > + If the loss of training data is small, then we inspect the loss on testing data. If the loss on testing data is small too, then you get a successful model, otherwise we still has to deal with it.
   >
   >   Following are the situations which may cause large loss on testing (validation) data:
   >
   >   + Overfitting: There are many ways to deal with this problem.
   >     + Use more training data. Some data augmentation techniques may be used. However, this is not the key point of machine learning.
   >     + Constrain model on its flexibility. To be more specific, use or choose less parameters or neurons, which highly demands background knowledges.
   >     + Early stopping.
   >     + Regularization.
   >     + Dropout.
   >   + Mismatch: Training data and testing data have different distribution. A lecture will focus on and discuss about this topic.
   >
   > + Finally, why do we need validation test?
   >
   >   Since we can not expose the testing data to our model, which is an action of cheating, we have to split a part of training data to replace the position of testing data.
   >   
   >   Sometimes overfitting can't be avoided even if validation test is used. This is because validation set is a "bad" set, which means it has a much different distribution from testing set. One solution is decreasing the number of training models.

## Lecture 2: Classification

