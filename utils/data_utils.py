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

### Show images function

# Define a function to show images from the dataset

def show_images(images, rescaled = False):
    num_images = len(images)
    rows = int(np.sqrt(num_images))
    cols = int(np.ceil(num_images / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    
    for i, ax in enumerate(axes.flat):
        image, label = images[i]
        image = np.transpose(image.numpy(), (1, 2, 0))
        
        if rescaled:
            image = (image * 255).astype(np.uint8) # rescale the image to [0, 255]
            
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(f"Label: {label}")
    
    plt.tight_layout()
    plt.show()  