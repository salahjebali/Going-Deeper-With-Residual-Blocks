### Start with some standard imports.
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from matplotlib.lines import Line2D
import pip

# Downlaoading the CIFAR10 dataset without transofrmations for the mean and std
cifar10 = CIFAR10(root='./data', train=True, download=True)

# Calculate the mean and std of the dataset
mean = cifar10.data.mean(axis=(0,1,2)) / 255.0
std = cifar10.data.std(axis=(0,1,2)) / 255.0

# Define the compute mean and std function
mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]

# Define the train and test transforms
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Donwload the dataset with the defined transformations
ds_train_cifar = CIFAR10('./data', train=True, download=True, transform=train_transform)
ds_test_cifar = CIFAR10('./data', train=False, download=True, transform=test_transform)

# Split train into train and validation
train_size = int(0.8 * len(ds_train_cifar))
val_size = len(ds_train_cifar) - train_size
ds_train_cifar, ds_val_cifar = torch.utils.data.random_split(ds_train_cifar, [train_size, val_size])

# Create dataloaders for train, validation and test (maybe in the future i will move this code to the main file)
batch_size = 64
train_loader_cifar = DataLoader(ds_train_cifar, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader_cifar = DataLoader(ds_val_cifar, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
test_loader_cifar = DataLoader(ds_test_cifar, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)


