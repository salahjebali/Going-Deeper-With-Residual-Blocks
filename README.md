# Going-Deeper-With-Residual-Blocks

In this repository the goal was to analyze the behavior of neural networks with respect to their depth. 
First I implemented a simple  `MLP ` for **image classification** task on MNIST dataset. 
Then, I implemented three CNNs model in [from here]([https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1409.1556)), `VGG16 `,  `VGG19 `, and a special version,  `VGG24 ` for testing the behavior of the neural networks going deeper, for image classification on CIFAR10 dataset. Then I implemented  `ResNet18 ` and  `ResNet34 ` [from here](https://arxiv.org/abs/1512.03385) for the same task and compared the two behaviors. 
Lastly, I tried to give a motivation of the behavior of  `residual block ` by studying the **gradients** and the **parameters**. 

Inside the `notebook` folder you can find a detailed **python notebook** of the project.

In the README.md, instead, it is possible to find a detailed analysis of the results obtained.

