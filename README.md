# Going-Deeper-With-Residual-Blocks

In this repository the goal was to analyze the behavior of neural networks with respect to their depth. 
First I implemented a simple  `MLP ` for **image classification** task on MNIST dataset. 
Then, I implemented three CNNs model in [from here]([https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1409.1556)), `VGG16 `,  `VGG19 `, and a special version,  `VGG24 ` for testing the behavior of the neural networks going deeper, for image classification on CIFAR10 dataset. Then I implemented  `ResNet18 ` and  `ResNet34 ` [from here](https://arxiv.org/abs/1512.03385) for the same task and compared the two behaviors. 
Lastly, I tried to give a motivation of the behavior of  `residual block ` by studying the **gradients** and the **parameters**. 

Inside the `notebook` folder you can find a detailed **python notebook** of the project.

In the README.md, instead, it is possible to find a detailed analysis of the results obtained.




# Scripts

In the [scripts](https://github.com/salahjebali/Going-Deeper-With-Residual-Blocks/tree/main/scripts) folder, you can find a script that contains several classes and methods utilized by all the model architectures for the Deep Learning Applications Labs - Lab 1.

The code inside the script has responsability of dealing with the following tasks

- **Model Classes**: Contains classes representing different model architectures.
- **Training**: Includes methods for training the models.
- **Evaluate**: Provides functions for evaluating model performance.
- **Weights Initialization**: Contains functions for initializing model weights.
- **Gradient Flow Analysis**: Includes utilities for analyzing gradient flow in models.
- **Image Display**: Contains methods to display images and visualizations.

Feel free to explore the .py file for understanding how it works.


# Exercise 1.1: A baseline MLP

Implement a simple Multilayer Perceptron to classify the 10 digits of MNIST (e.g. two narrow layers).Train this model to convergence, monitoring (at least) the loss and accuracy on the training and validation sets for every epoch.
Note: This would be a good time to think about abstracting your model definition, and training and evaluation pipelines in order to make it easier to compare performance of different models.

Important: Given the many runs you will need to do, and the need to compare performance between them, this would also be a great point to study how Tensorboard or Weights and Biases can be used for performance monitoring.# Your code here.

## 1.1.1: Convergence Study 

The convergence of the training and validation curves suggests that the model is effectively learning the underlying patterns present in the MNIST images. The similar trends between the training and validation curves indicate that the model is not suffering from overfitting.

![Validation adn Test curves](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/1.1%20MLP/val_train_plot.png)

## 1.1.2: Gradient Flow Study 

The observation that gradients appear to be stable and well-behaved during the training of your MLP for image classification on MNIST is a positive sign of effective learning and optimization.

![Gradient Flow](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/1.1%20MLP/gradient_flow.png)

# Exercise 1.2: Rinse and Repeat

Repeat the verification you did above, but with **Convolutional** Neural Networks. If you were careful about abstracting your model and training code, this should be a simple exercise. Show that **deeper** CNNs *without* residual connections do not always work better and **even deeper** ones *with* residual connections.

To conduct these analysis I used [Weight and Biases](https://wandb.ai/site) for tracking the gradients, the parameters and the convergence.

## 1.2.1: Compare all the models 

1.   **Research Question:**
    How do different models, including both ResNet and VGG architectures, perform in terms of accuracy on the validation set?
2. **Obtained Results:**
    After conducting several runs and plotting the validation and training curves, we observed the following performance ranking (from highest to lowest validation accuracy):


    1.   ResNet34
    2.   ResNet18
    3.   VGG19
    4.   VGG16
    5.   VGG24

3. **Interpretation:**
    From the results of Experiment 1, it is evident that the ResNet models outperform the VGG models consistently across all depths. This suggests that the residual connections in ResNet are significantly aiding in the learning process and preventing accuracy degradation as the network gets deeper. Additionally, we notice that VGG24 performs the worst among all models, indicating that increasing the depth of the VGG network has led to accuracy degradation.


**Train Accuracy**
![TA](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/1.2%20ResNet_vs_VGG/wandb/all_train_acc.png)
**Train Loss**
![TL](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/1.2%20ResNet_vs_VGG/wandb/all_train_loss.png)
**Val Accuracy** 
![VA](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/1.2%20ResNet_vs_VGG/wandb/all_val_accuracy.png)
**Val Loss**
![VL](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/1.2%20ResNet_vs_VGG/wandb/all_val_loss.png)

## 1.2.2: Compare ResNets architectures

1.   **Research Question:**
    Does increasing the depth of the ResNet models (ResNet18 vs. ResNet34) have any significant impact on the accuracy?
2. **Obtained Results:**
    After comparing the validation and training curves of ResNet18 and ResNet34, we found that both models achieved high validation revealsaccuracy. The performance ranking is as follows:


    1.   ResNet34
    2.   ResNet18

3. **Interpretation:**
    Experiment 3 reveals that increasing the depth from ResNet18 to ResNet34 has indeed improved the model's performance. This finding aligns with the inherent advantage of residual connections in ResNet, allowing it to leverage deeper architectures effectively without suffering from accuracy degradation. The performance gain observed in ResNet34 demonstrates the importance of adding more layers to enhance the model's capacity to learn complex patterns.

**Train Accuracy**
![TA](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/1.2%20ResNet_vs_VGG/wandb/resnet_train_acc.png)
**Train Loss**
![TL](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/1.2%20ResNet_vs_VGG/wandb/resnet_train_loss.png)
**Val Accuracy** 
![VA](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/1.2%20ResNet_vs_VGG/wandb/resnet_val_acc.png)
**Val Loss**
![VL](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/1.2%20ResNet_vs_VGG/wandb/resnet_val_loss.png)

## 1.2.3: Compare VGGs architectures

1.   **Research Question:**
    How does the accuracy of VGG models change as we increase the depth (VGG16 vs. VGG19 vs. VGG24)?
2. **Obtained Results:**
    Upon analyzing the validation and training curves of VGG16, VGG19, and VGG24, we observed the following performance ranking:

    1.   VGG16
    2.   VGG19
    3.   VGG24

3. **Interpretation:**
    Experiment 3 indicates that, unlike ResNet, increasing the depth of the VGG models negatively impacts accuracy. The highest accuracy was achieved by the shallower VGG16 model, followed by VGG19 and VGG24. This degradation in performance as the model deepens is likely due to the vanishing gradient problem, which is more pronounced in VGG architectures, limiting the model's capacity to learn effectively from deeper layers.

   
**Train Accuracy**
![TA](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/1.2%20ResNet_vs_VGG/wandb/vgg_train_acc.png)
**Train Loss**
![TL](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/1.2%20ResNet_vs_VGG/wandb/vgg_train_loss.png)
**Val Accuracy** 
![VA](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/1.2%20ResNet_vs_VGG/wandb/vgg_val_acc.png)
**Val Loss**
![VL](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/1.2%20ResNet_vs_VGG/wandb/vgg_val_loss.png)


# Exercise 2.1: Explain why Residual Connections are so effective

Use your two models (with and without residual connections) you developed above to study and quantify why the residual versions of the networks learn more effectively.

Hint: A good starting point might be looking at the gradient magnitudes passing through the networks during backpropagation.

## 2.1.0: Summary of the results

**1) Research Question**

   Why does ResNet34 outperform VGG16 in terms of accuracy? How do the gradients and parameters behave in both architectures?

**2) Obtained Results**


1.   **Gradients in ResNet**:

 The gradients in ResNet34 are stable and consistent at different layers, even in the deep layers. They do not suffer from vanishing gradients, which is crucial for successful training in deep networks.

2.   **Gradients in VGG**:

 The gradients in VGG16, especially in deeper layers, tend to be less stable and can sometimes suffer from the vanishing gradient problem. This instability may hinder the learning process and result in slower convergence.

3.  **Parameters in ResNet**:

 The parameters in ResNet34 are relatively stable throughout different layers. The stability of parameters suggests that the model is learning effectively, and there are no significant issues with the optimization process.

4. **Parameters in VGG**:

 In VGG16, the parameters show more sparsity, and in certain layers, they tend to diverge. The sparsity indicates that certain connections may not be contributing effectively to the model's learning. The divergence in parameters suggests instability in optimization and may hinder the model's convergence.

**3) Interpretation**

1.  **Stable Gradients in ResNet**:

 The stable gradients in ResNet34 enable efficient backpropagation of errors through the network, allowing the model to effectively learn from the data, even in deeper layers. This stability is a key factor in the success of ResNet.

2. **Vanishing Gradients in VGG**:

 The less stable gradients in VGG16, especially in deeper layers, indicate a higher likelihood of vanishing gradients. This phenomenon makes it challenging for the model to learn meaningful representations in the deeper layers and leads to slower convergence.

3. **Parameter Stability in ResNet**:

 The stable parameters in ResNet34 suggest that the model is consistently learning from the data and converging towards the optimal solution. This stability is crucial for the model to make meaningful adjustments during training.

4. **Parameter Sparsity and Divergence in VGG**:

  The parameter sparsity in VGG16 implies that certain connections may not be effectively contributing to the learning process, limiting the model's capacity to capture complex patterns. The divergence in parameters indicates instability in the optimization process, which can hinder the model's ability to converge and generalize well.


  In summary, the superior performance of ResNet34 over VGG16 can be attributed to the stability of gradients and parameters in ResNet, allowing it to effectively handle deeper architectures without suffering from accuracy degradation. The vanishing gradients and parameter instability in VGG16 impede its ability to learn efficiently in deeper layers, leading to decreased accuracy.


## 2.1.1: Gradient Analysis

### 2.1.1.0: VGG
The observation of gradients approaching zero and being very sparse in VGG is concerning and points to the presence of the vanishing gradient problem. When gradients become too small, they effectively diminish as they propagate backward through the layers during training. As a result, layers towards the beginning of the network may not receive sufficiently informative updates, hindering their ability to learn and adapt to the data. Consequently, this vanishing gradient issue can severely limit the capacity of VGG to capture complex features and may lead to decreased accuracy, particularly in deeper layers.

The presence of very sparse gradients further exacerbates the vanishing gradient problem in VGG. Sparse gradients suggest that certain connections within the network are rarely updated during training. This sparsity can prevent crucial information from being effectively propagated throughout the network, causing information loss and impeding the learning process. Sparse gradients can also result in longer convergence times, as the model may require more iterations to adapt its parameters properly.


![TA](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/2.1%20ResNet_can_go_deeper/vgg/gradients/gradients_features.49.weight.png)

![VL](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/2.1%20ResNet_can_go_deeper/vgg/gradients/gradients_classifier.3.weight.png)

![TL](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/2.1%20ResNet_can_go_deeper/vgg/gradients/gradients_features.8.weight.png)

![VA](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/2.1%20ResNet_can_go_deeper/vgg/gradients/gradients_features.8.bias.png)

### 2.1.1.1: ResNet

The observed uniform distribution of gradients around zero in ResNet is a highly favorable characteristic that indicates the network's resilience against both vanishing and exploding gradient problems. The fact that the gradients are not sparse implies that all connections in the network are effectively updated during backpropagation, ensuring that information flows consistently through the layers. This uniformity is crucial for successful training, as it allows the model to learn meaningful representations across all depths.

The absence of vanishing gradients is a significant advantage in ResNet. When gradients remain consistently non-zero and do not approach extremely small values, the network can efficiently backpropagate errors through the layers. This property enables the model to learn from the data effectively and efficiently, even in very deep architectures.

Similarly, the lack of exploding gradients ensures that the optimization process remains stable. When gradients are not prone to extreme values, the risk of the optimization process becoming unstable and leading to drastic parameter updates is mitigated. A stable optimization process helps in achieving faster convergence and better generalization.

The uniform distribution of gradients around zero in ResNet, along with the absence of sparsity and exploding gradients, is a testament to the effectiveness of the residual connections in the architecture. These skip connections allow for the smooth propagation of gradients, fostering consistent learning throughout the network. As a result, ResNet can handle deeper architectures without suffering from accuracy degradation, making it a superior choice for various deep learning tasks.


![TA](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/2.1%20ResNet_can_go_deeper/resnet/gradients/gradients_layer2.0.conv1.weight.png)
![TL](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/2.1%20ResNet_can_go_deeper/resnet/gradients/gradients_layer1.2.conv1.weight.png)
![VA](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/2.1%20ResNet_can_go_deeper/resnet/gradients/gradients_layer2.2.conv2.weight.png)
![VL](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/2.1%20ResNet_can_go_deeper/resnet/gradients/gradients_layer3.3.conv1.weight.png)

## 2.1.2: Parameter Analysis

### 2.1.2.0: VGG 
The observed sparsity of parameters in VGG is an interesting finding. Sparse parameters suggest that certain connections in the network are not effectively contributing to the learning process, which might be limiting the model's capacity to capture complex patterns in the data. Sparse connections can lead to information loss and hinder the model's ability to represent fine-grained details in the input. Additionally, the noise present during convergence in VGG can be indicative of instability in the optimization process. This instability might lead to fluctuations in parameter values, making it difficult for the model to converge smoothly to an optimal solution.


![TA](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/2.1%20ResNet_can_go_deeper/vgg/parameters/parameters_features.10.bias.png)

![TL](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/2.1%20ResNet_can_go_deeper/vgg/parameters/parameters_features.11.weight_features.11.weight.png)

![VA](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/2.1%20ResNet_can_go_deeper/vgg/parameters/parameters_features.21.bias.png)

![VL](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/2.1%20ResNet_can_go_deeper/vgg/parameters/parameters_features.37.weight.png)

### 2.1.2.1: ResNet
In contrast, ResNet showcases remarkable stability in both its parameters and gradients, which is a positive sign for deep neural networks. The stability of parameters in ResNet indicates consistent learning from the data and convergence towards an optimal solution. The absence of significant fluctuations or sparsity in parameters signifies that ResNet is utilizing all connections effectively to represent complex patterns. The improved stability in ResNet can be attributed to its unique architecture with residual connections, which mitigate the vanishing gradient problem and encourage more straightforward optimization. The stability observed in ResNet is a significant advantage, as it contributes to its ability to perform well even with very deep networks, avoiding accuracy degradation and enhancing its generalization capabilities.


![TA](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/2.1%20ResNet_can_go_deeper/resnet/parameters/parameters_layer3.4.conv2.weight.png)
![TL](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/2.1%20ResNet_can_go_deeper/resnet/parameters/parameters_layer3.2.conv1.weight.png)
![VA](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/2.1%20ResNet_can_go_deeper/resnet/parameters/parameters_layer3.0.bn1.weight.png)
![VL](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/2.1%20ResNet_can_go_deeper/resnet/parameters/parameters_layer2.0.residual.conv.weight.png)
