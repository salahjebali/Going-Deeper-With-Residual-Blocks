### Start with some standard imports.
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import torch
import torch.optim as optim
from torch.optim import SGD
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import pip

from data import train_loader_cifar, val_loader_cifar, test_loader_cifar
from models import VGG
from scripts import TrainerManager
from utils import resnet

in_channels = 3
num_classes = 10
input_size = 784 # assuming input image size is 28x28

# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# VGG16 model
vgg16_architecture = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
vgg_16 = VGG(in_channels, num_classes, vgg16_architecture)
# vgg_16.to(device) # move the model to GPU

# ResNet18 model
resnet_18 = resnet('resnet18', num_classes, pretrained = False)

### Training setting 

# Hyperparameters
hyperparams = {
    "learning_rate": 0.01,
    "weight_decay": 0.0001,
    "momentum": 0.9,
    "num_epochs": 50,
}

run_config_resnet18 = {
    "learning_rate": hyperparams["learning_rate"],
    "weight_decay": hyperparams["weight_decay"],
    "momentum": hyperparams["momentum"],
    "optimizer": "SGD",
    "architecture": "ResNet18",
    "dataset": "CIFAR10",
    "epochs": hyperparams["num_epochs"],
}

run_config_vgg16 = {
    "learning_rate": hyperparams["learning_rate"],
    "weight_decay": hyperparams["weight_decay"],
    "momentum": hyperparams["momentum"],
    "optimizer": "SGD",
    "architecture": "VGG16",
    "dataset": "CIFAR10",
    "epochs": hyperparams["num_epochs"],
}

# Optimizer
resnet_18_optimizer = optim.SGD(resnet_18.parameters(), lr = hyperparams["learning_rate"], momentum = hyperparams["momentum"], weight_decay = hyperparams["weight_decay"])
vgg_16_optimizer = optim.SGD(vgg_16.parameters(), lr = hyperparams["learning_rate"], momentum = hyperparams["momentum"], weight_decay = hyperparams["weight_decay"])

# Scheduler
resnet_18_scheduler = MultiStepLR(resnet_18_optimizer, milestones=[int(0.5 * hyperparams["num_epochs"]), int(0.75 * hyperparams["num_epochs"])], gamma=0.1)
vgg_16_scheduler = MultiStepLR(vgg_16_optimizer, milestones=[int(0.5 * hyperparams["num_epochs"]), int(0.75 * hyperparams["num_epochs"])], gamma=0.1) 

# Loss function
criterion = nn.CrossEntropyLoss()

# Training loop
global_trainer = TrainerManager(device)

# ResNet18 training
global_trainer.reset_training_history()
global_trainer.train(resnet_18,
                     train_loader_cifar,
                     val_loader_cifar,
                     criterion, 
                     resnet_18_optimizer, 
                     num_epochs = hyperparams["num_epochs"],
                     gradient_flow = False, 
                     scheduler = resnet_18_scheduler,
                     run_config = run_config_resnet18)

# VGG16 training
global_trainer.reset_training_history()
global_trainer.train(vgg_16,
                     train_loader_cifar,
                     val_loader_cifar,
                     criterion, 
                     vgg_16_optimizer, 
                     num_epochs = hyperparams["num_epochs"],
                     gradient_flow = False, 
                     scheduler = vgg_16_scheduler,
                     run_config = run_config_vgg16)