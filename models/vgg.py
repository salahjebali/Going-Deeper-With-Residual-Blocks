### Start with some standard imports.
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import torch
import torch.nn as nn



from utils import *

class VGG(BasicModel):
    
    def __init__(self, in_channels, num_classes, architecture):
        """
        VGG model

        Args:
            in_channels (int): channels of the input
            num_classes (int): channels of the output
            architecture (array): array of integers representing the architecture of the model
        """
        super(VGG, self).__init__()
        
        self.features = self._make_layers(in_channels, architecture)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        
        self.apply(init_weights)
        
    def _make_layers(self, in_channels, architecture):
        """
        Make layers of the model

        Args:
            in_channels (int): Number of input channels
            architecture (array): array of integers representing the architecture of the model

        Returns:
            nn.Sequential: layers of the model
        """
        layers = []
        for x in architecture:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
            else:
                out_channels = x
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = x
        
        return nn.Sequential(*layers)