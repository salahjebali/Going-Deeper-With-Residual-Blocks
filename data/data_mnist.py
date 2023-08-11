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

# Standard MNIST transform

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

# Load MNIST dataset train and test sets
ds_train = MNIST('./data', train=True, download=True, transform=transform)
ds_test = MNIST('./data', train=False, download=True, transform=transform)

# Split train inot train and validation
val_size = 5000
I = np.random.permutation(len(ds_train))
ds_val = Subset(ds_train, I[:val_size])
ds_train = Subset(ds_train, I[val_size:])

# Create dataloaders for train, validation and test (maybe in the future i will move this code to the main file)
batch_size = 64
train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
