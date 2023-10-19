"""
Utility functions for training
"""

# Import Libraries
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import subprocess

# Import Modules
from src.utils import get_project_root


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
    
def start_tensorboard():
    log_dir = os.path.join(get_project_root(), 'logs/tensorboard')
    tensorboard_cmd = f"tensorboard --logdir={log_dir}"
    writer = SummaryWriter(log_dir)
    process = subprocess.Popen(tensorboard_cmd.split())
    return process, writer

def stop_tensorboard(tensorboard_process):
    if tensorboard_process:
        tensorboard_process.terminate()
        
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