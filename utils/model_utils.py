### Start with some standard imports.
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import torch
import torch.optim as optim
from torch.optim import SGD
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import Subset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from matplotlib.lines import Line2D
from torch.optim.lr_scheduler import MultiStepLR
import pip

from models import resnet

### BasicModel class

class BasicModel(nn.Module):
    """
    Basic model class

    Args:
        nn (_type_): _description_
    """
    
    def ___init__(self):
        super(BasicModel, self).__init__()
        
    def forward(self, x):
        """
        Forward pass of the model
        
        Args:
            x (_type_): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError("Forward pass not implemented")
    
    def count_parameters(self):
        """
        Counts the number of parameters in the model

        Returns:
            _type_: _description_
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self):
        """
        Prints a summary of the model
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Model summary")
        print("-"*50)
        print(self)
        print(f"Trainable parameters: ", params)
        print("-"*50)
        
        
### Init weights function 

def init_weights(module):
    """
    Initialize the weights of the model.

    Args:
        module (torch.nn.Module): The module whose weights we want to initialize.
    """
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        if module.weight is not None:
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)
    elif isinstance(module, nn.BatchNorm2d):
        if module.weight is not None:
            nn.init.constant_(module.weight.data, 1)
        if module.bias is not None:
           nn.init.constant_(module.bias.data, 0)
           
### Function for defining the ResNet model in function of the depth
def resnet(name, num_classes=10, pretrained=False):
    """
    Returns suitable ResNet model from its name.

    Args:
        - name (str): name of resnet architecture.
        - num_classes (int): number of target classes.
        - pretrained (bool): whether to use a pretrained model.

    Returns:
        - torch.nn.Module.
    """

    if name == 'resnet18':
        return ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=num_classes)
    elif name == 'resnet34':
        return ResNet(ResidualBlock, [3, 4, 6, 3], num_classes=num_classes)
    raise ValueError('Only resnet18, resnet34 are supported!')
           