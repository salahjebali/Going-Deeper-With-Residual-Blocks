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
# Make sure wandb is installed
import subprocess 
import sys

### Configuration of weights and biases 

# Ensure deterministi behaviour 

torch.backends.cudnn.deterministic = True
np.random.seed(hash("setting sandom seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 -1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)


# Install wandb using pip   
# %%capture
# !pip install wandb --upgrade
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'wandb'])

# Import wandb and login
import wandb

wandb.login()

### Analyzing gradient flow 

def plot_grad_flow(named_parameters):
    """
    Plots the gradients flowing through different layers in the net during training.
    Usage: Plug this function in Trainer class after loss.backward() as plot_grad_flow(self.model.named_parameters()) to visualize the gradient flow.

    Args:
        named_parameters (_type_): _description_
    """
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.grad is not None:
            if p.requires_grad and 'bias' not in n:
                layers.append(n)
                if p.grad.is_cuda:
                    grad = p.grad.cpu().abs()
                else:
                    grad = p.grad.abs()
                ave_grads.append(grad.mean())
                max_grads.append(grad.max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

### Trainer class

# Define the Trainer class containing the training and evaluation loops with weights and biases logging

class TrainerManager():
    """
    Trainer class for training and evaluation loops with weights and biases logging.
    """
    
    def __init__(self, device='cpu'):
        """
        Trainer class

        Args:
            device (str, optional): device. Defaults to 'cpu'.
        """
        self.device = device
        
        # Trainig history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train(self, model, train_loader, val_loader, criterion, optimizer, num_epochs, gradient_flow = False, scheduler = None, run_config = None):
        """
        Training loop

        Args:
            model (nn.Module): model to train
            train_loader (DataLoader): data loader for the training set
            val_loader (DataLoader): data loader for the validation set
            criterion (torch.nn): loss function
            optimizer (torch.nn): optimizer
            num_epochs (int): number of epochs
            gradient_flow (bool, optional): flag to enable gradient flow logging. Defaults to False.
            scheduler (torch.nn, optional): scheduler. Defaults to None.
            run_config (dict, optional): dictionary containing run config for wandb. Defaults to None.
        """
        model.to(self.device)


        # 1) Start a new run to track this script
        wandb.init(project="lab 1 - dla", config=run_config)        
        
        # 2) Copy the config to avoid any reference issues
        config = wandb.config
        
        # 3) Wathc the model for gradients tracking
        wandb.watch(model, log="all")
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            for batch, (X,y) in enumerate(train_loader):
                X = X.to(self.device)
                y = y.to(self.device)
                
                # Forward pass
                y_hat = model(X)
                loss = criterion(y_hat, y)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                
                # Flag to decide whether to plot the gradient flow
                if gradient_flow:
                    plot_grad_flow(model.named_parameters())

                optimizer.step()
                
                # Track the loss
                running_loss += loss.item() * X.size(0)
                
                # Track the accuracy
                _, predicted = torch.max(y_hat.data, 1)
                total_predictions += y.size(0)
                correct_predictions += (predicted == y).sum().item()
                
            # Calculate the average loss and accuracy over the entire epoch
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_accuracy = correct_predictions / total_predictions
            
            # Store the training epoch results
            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_accuracy)
            
            # 4) Train metrics for wandb
            
            train_metrics = {
                "train/train_loss": epoch_loss,
                "train/train_accuracy": epoch_accuracy,
                "train/epoch": epoch
            }
            
            # Print epoch statistics
            print(f"Epoch {epoch+1}/{num_epochs} - loss: {epoch_loss:.4f} - accuracy: {epoch_accuracy:.4f}")
            
            # Check if there is any scheduler to update
            if scheduler is not None:
                scheduler.step()
                
            # Perform validation
            val_loss, val_acc = self._validate(model, val_loader, criterion)
            
            # 5) Validation metrics for wandb
            
            val_metrics = {
                "val/val_loss": val_loss,
                "val/val_accuracy": val_acc,
            }
            
            # 6) Log metrics to wandb
            wandb.log({**train_metrics, **val_metrics})
            
            # Save the model checkpoint if the validation loss is the best we've seen so far 
            if epoch == 0 or val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pth')
                
            # 8) Save the model checkpoint to wandb
            wandb.save('best_model.pth')
            
            # 9) Mark the run as finished
            wandb.finish()
            
    def _validate(self, model, val_loader, criterion):
        
        """
        Performs the validation loop
        
        Args:
            model (nn.Module): model to evaluate
            val_loader (DataLoader): validation data loader
            criterion (_type_): loss function

        Returns:
            val_loss (float): average loss over the validation set
            val_accuracy (float): average accuracy over the validation set
        """
        model.to(self.device)
        model.eval()
        val_loss = 0.0
        val_correct_predictions = 0
        val_correct_predictions = 0
        
        with torch.no_grad():
            for batch, (X, y) in enumerate(val_loader):
                X = X.to(self.device)
                y = y.to(self.device)
                
                # Forward pass
                y_hat = model(X)
                loss = criterion(y_hat, y)
                
                # Track the loss
                val_loss += loss.item() * X.size(0)
                
                # Track the accuracy
                _, predicted = torch.max(y_hat.data, 1)
                val_total_predictions += y.size(0)
                val_correct_predictions += (predicted == y).sum().item()
                
        # Calculate the average loss and accuracy over the entire validation set
        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = val_correct_predictions / val_total_predictions
        
        # Store the validation epoch results
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_accuracy)
        
        # Print statistics
        print(f"Validation loss: {val_loss:.4f} - validation accuracy: {val_accuracy:.4f}")
        
        return val_loss, val_accuracy
    
    def test(self, model, test_loader):
        """
        Evaluate the model on the testing dataset and print the test accuracy.


        Args:
            - test_loader (torch.utils.data.DataLoader): DataLoader for the testing dataset.
        """

        model.to(self.device)
        model.eval()  # Set the model in evaluation mode
        test_correct_predictions = 0
        test_total_predictions = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total_predictions += labels.size(0)
                test_correct_predictions += (predicted == labels).sum().item()

        # Calculate test accuracy
        test_accuracy = test_correct_predictions / test_total_predictions

        # Print test accuracy
        print(f"Test Accuracy: {test_accuracy:.4f}")
        return test_accuracy
    
    def plot_curves(self):
        """
        Plot the training and validation curves.
        """
        epochs = len(self.train_losses)
        x = range(1, epochs + 1)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(x, self.train_losses, label="Training Loss")
        plt.plot(x, self.val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(x, self.train_accuracies, label="Training Accuracy")
        plt.plot(x, self.val_accuracies, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.show()


    def reset_training_history(self):
        """
        This function is called to reset training history
        when you want to apply the trainer to a different model
        """

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
