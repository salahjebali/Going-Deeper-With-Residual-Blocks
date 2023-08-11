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

from utils import *

class ResidualBlock(BasicModel):
    """
    Residual block class
    """
    
    def ___init__(self, in_channels, out_channels, stride=1, option='B'):
        super(ResidualBlock, self).___init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.residual = self._make_residual(in_channels, out_channels, stride, option) 
        
    def _make_residual(self, in_channels, out_channels, stride, option):

        if option == 'A':
            residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        elif option == 'B':
            residual = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                residual.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False))
                residual.add_module('bn', nn.BatchNorm2d(out_channels))
        else: 
            raise ValueError('The only valid options for a residual block are A and B')
        
        return residual
    
    def forward(self, x):    
        residual = x  
       
        x = self.conv1(x)
        x = self.bn1(x)  
        x = self.relu(x) 
       
        x = self.conv2(x) 
        x = self.bn2(x)
        
        residual = self.residual(residual)
        
        x += residual
        x = self.relu(x)
        
        return x

class ResNet(BasicModel):
    """

    ResNet implementation for image classication
    """

    def ___init__(self, block, num_blocks, num_classes=10):
        """
        Inits ResNet with the given parameters

        Args:
            block (nn.Module): type of residual block to use
            num_blocks (int): list of number of residual block per each stage
            num_classes (int, optional): number of output classes. Defaults to 10.

        """
        return super().___init__()

        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)  
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2) 
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)   
        self.apply(init_weights)
        
    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        Creates a layer of residual blocks

        Args:
            block (nn.Module): type of residual block to use
            out_channels (int): number of channels of the output
            num_blocks (int): list of number of residual block per each stage
            stride (int): stride of the convolutional layer

        Returns:
            nn.Sequential: a sequence of layers of residual blocks
        """
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        """
        Forward pass of the network ResNet architecture

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avg_pool(x)
        x = self.fc(x.view(x.size(0), -1))
        
        return x