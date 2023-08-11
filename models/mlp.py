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

class MLPModel(BasicModel):
    
    def __init__(self, input_size, hidden_size, num_classes,):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.apply(init_weights)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
