"""
Utility functions for training
"""

# Import Libraries
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import subprocess
from tqdm.autonotebook import tqdm

# Import Modules
from src.utils import get_project_root


class TrainingState:
    def __init__(self):
        
        self.train_losses = []
        self.val_losses = []
        self.epoch = 0
        self.elapsed_time = 0
        self.train_elapsed_time = 0
        self.mean_loss_epoch = 0
        self.mean_loss_training = 0
        self.mean_loss_val = 0
        self.val_loss = 0
        
    def update(self, **kwargs):
        for attr, value in kwargs.items():
            if hasattr(self, attr):
                if attr == "mean_loss_epoch":
                    self.train_losses.append(value)
                    self.mean_loss_training = np.mean(self.train_losses)
                if attr == "val_loss":
                    self.val_losses.append(value)
                    self.mean_loss_val = np.mean(self.val_losses)
                if attr == "elapsed_time":
                    self.train_elapsed_time += value
                setattr(self, attr, value)
            else:
                raise AttributeError(f"{self.__class__.__name__} object has no attribute {attr}")

class EarlyStopping:
    def __init__(self, no_change_patience, overfit_patience, warmup, verbose=False):
        self.best_val_loss = float('inf')
        self.no_change_count = 0
        self.overfit_count = 0
        self.no_change_patience = no_change_patience
        self.overfit_patience = overfit_patience
        self.warmup = warmup
        self.verbose = verbose

    def should_stop(self, state):
        stop = False
        new_best = False
        if state.val_loss < self.best_val_loss:
            self.best_val_loss = state.val_loss
            self.no_change_count = 0
            new_best = True
        else:
            self.no_change_count += 1

        if state.mean_loss_val > state.mean_loss_epoch:
            self.overfit_count += 1
        else:
            self.overfit_count = 0

        if state.epoch >= self.warmup:
            if self.no_change_count >= self.no_change_patience:
                stop = True
                reason = "no improvement in validation loss"
            elif self.overfit_count >= self.overfit_patience:
                stop = True
                reason = "overfitting"
                
        if stop:
            if self.verbose:
                print(f"Early stopping due to {reason}.")
            else:
                pass
        
        return stop, new_best
    
class TensorBoardLogger:
    def __init__(self, use_tensorboard):
        self.writer = None
        self.process = None
        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard:
            self.start()

    def start(self):
        """Start the TensorBoard process and SummaryWriter"""
        if self.use_tensorboard:
            log_dir = os.path.join(get_project_root(), 'logs/tensorboard')
            tensorboard_cmd = f"tensorboard --logdir={log_dir}"
            self.writer = SummaryWriter(log_dir)
            self.process = subprocess.Popen(tensorboard_cmd.split())
        
    def log_loss(self, state):
        """Log the training loss."""
        if self.use_tensorboard:
            self.writer.add_scalars('Loss', {'Training Loss': state.mean_loss_epoch, 'Validation Loss': state.val_loss}, state.epoch)
        
    def log_params_grads(self, model, epoch):
        """Log model parameters and gradients."""
        if self.use_tensorboard:
            for name, param in model.named_parameters():
                self.writer.add_histogram(name, param, epoch)
                self.writer.add_histogram(f"{name}.grad", param.grad, epoch)

    def stop(self):
        """Terminate the TensorBoard process."""
        if self.process:
            self.process.terminate()
        if self.writer:
            self.writer.close()

def plot_data(actual, predicted, set_name, loss, column_to_plot, xlim=None, ylim=None):
    # Ensure column_to_plot is a list
    if not isinstance(column_to_plot, list):
        column_to_plot = [column_to_plot]
    
    num_subplots = len(column_to_plot)
    
    fig, axes = plt.subplots(nrows=num_subplots, ncols=1, figsize=(10, 3*num_subplots))
    
    if num_subplots == 1:
        axes = [axes]  # Make it iterable for consistency
    
    for idx, col in enumerate(column_to_plot):
        ax = axes[idx]
        
        if col == 'Close':
            plot_data = actual['Close'].shift(1) * (1 + predicted['percent_ret'])
            actual_data = actual[['Close']]
        elif col == 'percent_ret':
            plot_data = predicted
            actual_data = actual[['percent_ret']]
        else:
            raise ValueError(f"Unsupported column: {col}")
        
        ax.plot(actual_data.index, actual_data, label='Actual', color='green', lw=1)
        ax.plot(plot_data.index, plot_data, label='Predicted', color='red', lw=1)
        
        if xlim is not None and len(xlim) > 0:  # Set x-axis limit if provided
            xlim = pd.to_datetime(xlim)
            ax.set_xlim(xlim)
        if ylim:  # Set y-axis limit if provided
            ax.set_ylim(ylim)
        
        ax.set_title(f'{col} Over Time ({set_name}, MSE={loss:.5g})', fontsize=16)
        ax.set_xlabel('Time', fontsize=14)
        ax.set_ylabel(f'Value ({col})', fontsize=14)
        ax.legend()

    plt.tight_layout()
    plt.show()

def plot_learning_curves(train_losses, val_losses):
    """
    Plot the learning curves for training and validation losses.
    
    Parameters:
    - train_losses: List of training loss values
    - val_losses: List of validation loss values
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Plotting the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='x')
    
    plt.title('Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.show()
        
def save_model(model, filename):
    #TODO: save optimizer, loss, epoch...
    checkpoint_path = os.path.join(get_project_root(), f'checkpoints/model')
    model_path = os.path.join(checkpoint_path, f'{filename}.pth')
    torch.save(model.state_dict(), model_path)

def load_model(model, filename, verbose=False):
    """
    Load a pre-trained model from the given path.
    """
    checkpoint_path = os.path.join(get_project_root(), f'checkpoints/model')
    model_path = os.path.join(checkpoint_path, f'{filename}.pth')
    model.load_state_dict(torch.load(model_path))
    if verbose:
        print(f'Model loaded --------- {model_path}')
