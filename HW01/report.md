# Report of HW01

[TOC]

## Initial version

![image-20230224114630553](/home/carolt/snap/typora/76/.config/Typora/typora-user-images/image-20230224114630553.png)

## Version 01 ---- Complexer Model

Add more layers.

Neural network model before begin modified:

````python
class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions. 
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1) # (B, 1) -> (B)
        return x
````

Neural network model after being modified:

````python
class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions. 
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1) # (B, 1) -> (B)
        return x
````

> <img src="/home/carolt/SelfEducating/Artificial_Intelligence/ML2022-Spring/HW01/images/image-20230224122823241.png" alt="image-20230224122823241" style="zoom:80%;" />

Obviously, the model is still not complex enough.

Then I make the model more complex by adding another layer, but obtain few performance enhancement.

## Version 02 ---- Selecting Features

I deleted features about id and states, then I got result as following:

> <img src="/home/carolt/SelfEducating/Artificial_Intelligence/ML2022-Spring/HW01/images/image-20230226212055551.png" alt="image-20230226212055551" style="zoom:80%;" />
>
> Grey one is version 01 while BLUE one is version 02.

Much better, right?

## Version 03 ---- Smaller batch size

After setting a smaller batch size, I regretted. That is because actually my version 02 model didn't get a stable loss curve at the end of training, which means the model may not need more jilter. Maybe I should set a bigger batch size to speed the training up.

However, actually there is no big change in the result of training after I set a smaller batch size.

## Version 04 ---- Adaptive Learning Rate

I chose Exponential scheduler and added a few lines:

````python
# Creates my schedulers.
scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
````

````python
scheduler1.step() # Update learning rate.
````

And the result was the ORANGE line:

> <img src="/home/carolt/SelfEducating/Artificial_Intelligence/ML2022-Spring/HW01/images/image-20230227125247843.png" alt="image-20230227125247843" style="zoom:80%;" />

The training stopped early after using scheduler.

## Version 05 ---- L2 Regularization with Weight Decay

I added a parameter `weight_decay=0.01` in the SGD optimizer. And the result was the GREEN line:

> <img src="/home/carolt/SelfEducating/Artificial_Intelligence/ML2022-Spring/HW01/images/image-20230227132623238.png" alt="image-20230227132623238" style="zoom:80%;" />

No obvious improvement.

## Conclusion

By comparing the gray line to the others, measurements like adaptive learning rate and L2 regularization prevented the training from **overfitting**.