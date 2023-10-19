"""
Trainer class definition
"""

# Import Libraries
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchinfo import summary
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import subprocess


# TODO: 1. Move plotting functions to a separate utility module like /src/utils/plotting.py.
# TODO: 2. Move the predict function to a separate module like /predict/predictor.py.
# TODO: 3. Consider encapsulating TensorBoard logic into its own class or module.
# TODO: 4. Pass a configuration dictionary or object to the Trainer constructor to simplify parameter management.
# TODO: 5. Add docstrings to explain the purpose and functionality of each method.
# TODO: 6. Encapsulate early stopping logic into its own method.
# TODO: 7. Make variable names more descriptive for better readability (e.g., lr, no_change_patience, etc.).
# TODO: 8. Encapsulate model state saving and loading logic into separate methods or a separate class.
# TODO: 9. Allow the choice of optimizer and scheduler to be passed as arguments or included in a configuration object.


class Trainer:
    def __init__(self,
                 model,
                 device,
                 verbose=False,
                 running_in_colab=False,
                 use_TPU=False,
                 use_tensorboard=False):
        
        # Model parameters
        self.model = model
        self.model_name = self.model.__class__.__name__
        self.device = device
        self.criterion = nn.MSELoss()  # Mean Squared Error Loss
        
        # Debug
        self.verbose = verbose
        self.running_in_colab = running_in_colab
        self.use_TPU = use_TPU
        self.use_tensorboard = use_tensorboard

    def train_model(self, train_loader, val_loader, epochs, lr, no_change_patience, overfit_patience, warmup, save_best=False):
        # Training Parameters
        self.epochs = epochs
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(self.optimizer, 'min')
        self.no_change_patience = no_change_patience
        self.overfit_patience = overfit_patience
        self.model = self.model.to(self.device)
        self.warmup = warmup

        self.train_loader = train_loader
        # self.val_loader = val_loader

        if self.verbose:
            print(f'Training --------- Model: {self.model_name}')
            
        if self.use_tensorboard:
            self.tensorboard_process = self.start_tensorboard()

        train_losses = []
        val_losses = []
        
        train_elapsed_time = 0
        best_val_loss = float('inf')  # Initialize best validation loss
        no_improvement_count = 0  # Counter for early stopping
        overfit_count = 0
        mean_loss_epoch = 0
        mean_loss_training = 0
        mean_loss_val = 0
        val_loss = 0
        elapsed_time = 0

        tqdm_epochs = tqdm(range(epochs), disable=not self.verbose, desc='Epochs')
        
        for epoch in tqdm_epochs:
            self.model.train()
            batch_losses = []
            start_time = time.time()

            for i, (x, y) in enumerate(train_loader):
                tqdm_epochs.set_description_str(desc=f"Batch [{i+1}/{len(train_loader)}]\t")
                
                x, y = x.to(self.device), y.to(self.device)

                # Forward pass
                outputs = self.model(x).squeeze()  # Remove extra dimension from model's output
                loss = self.criterion(outputs, y)

                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_losses.append(loss.item())
                
                postfix_str = (
                    f"Batch Loss: {loss.item():.4g} ({np.mean(batch_losses):.4g})\t"
                    f"Loss: {mean_loss_epoch:.4g} ({mean_loss_training:.4g})\t"
                    f"Val Loss: {val_loss:.4g} ({mean_loss_val:.4g})"
                )
                tqdm_epochs.set_postfix_str(postfix_str)

            elapsed_time = time.time() - start_time
            train_elapsed_time += elapsed_time
            mean_loss_epoch = np.mean(batch_losses)
            train_losses.append(mean_loss_epoch)
            mean_loss_training = np.mean(train_losses)
            
            # Validation loop
            val_loss = self.evaluate_model(val_loader)
            val_losses.append(val_loss)
            mean_loss_val = np.mean(val_losses)
            scheduler.step(val_loss)    # Learning rate scheduler step
            
            if self.use_tensorboard:
                # Log scalar values - TensorBoard
                self.writer.add_scalar('Training Loss', mean_loss_epoch, epoch)
                self.writer.add_scalar('Validation Loss', val_loss, epoch)
                
                # Log model parameters and gradients - TensorBoard
                for name, param in self.model.named_parameters():
                    self.writer.add_histogram(name, param, epoch)
                    self.writer.add_histogram(f"{name}.grad", param.grad, epoch)
            
            # Early stopping logic for no improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_count = 0
                if save_best:
                    self.save_model(model_path=f"saved_models/{self.model_name}.pth")
            else:
                no_improvement_count += 1

            # Early stopping logic for overfitting
            if mean_loss_val > mean_loss_epoch:
                overfit_count += 1
            else:
                overfit_count = 0  # Reset if not overfitting
            
            if epoch >= self.warmup:
                if no_improvement_count >= no_change_patience or overfit_count >= overfit_patience:
                    reason = "no improvement in validation loss" if no_improvement_count >= no_change_patience else "overfitting"
                    print(f"Early stopping due to {reason}.")
                    break
            
        # Save learning data
        self.train_losses = train_losses
        self.val_losses = val_losses
        if self.verbose:
            print(f'Total Train Time: {train_elapsed_time:.2f}s, Avg Epoch: {train_elapsed_time/epochs:.2f}s')
        
        # Close TensorBoard writer
        if self.use_tensorboard:
            self.writer.close()

    def evaluate_model(self, loader):
        self.model.eval()  # Set model to evaluation mode
        start_time = time.time()

        # if self.verbose:
            # print(f'Evaluating {set_name} Set --------- Model: {self.model_name}')
        
        # t = tqdm(loader, disable=not self.verbose)    # Progress bar

        total_loss = 0.0
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)

                outputs = self.model(x).squeeze()  # Forward pass

                loss = self.criterion(outputs, y)  # Compute loss

                total_loss += loss.item()
                
                # if self.verbose:
                #     t.set_description(f'[{i+1}/{len(loader)}] Loss: {total_loss/(i+1):.4f}')
            
        elapsed_time = time.time() - start_time

        mean_loss = total_loss / len(loader)

        # if self.verbose:
        #     t.set_description(f'Loss: {mean_loss:.4f}, Time Elapsed: {elapsed_time:.2f}s')

        return mean_loss

    def plot_data(self, actual, predicted, set_name, loss, column_to_plot, xlim=None, ylim=None):
        
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

    def plot_learning_curves(self, train_losses, val_losses):
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
        
    def start_tensorboard(self):
        log_dir = 'tensorboard/experiment_1'
        tensorboard_cmd = f"tensorboard --logdir={log_dir}"
        self.writer = SummaryWriter(log_dir)
        process = subprocess.Popen(tensorboard_cmd.split())
        return process

    def stop_tensorboard(self):
        if self.tensorboard_process:
            self.tensorboard_process.terminate()
            
    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)
    
    def load_model(self, model_path):
        """
        Load a pre-trained model from the given path.
        
        Parameters:
        - model_path (str): The path to the saved model's state dictionary.
        """
        self.model.load_state_dict(torch.load(model_path))
        if self.verbose:
            print(f'Model loaded --------- {model_path}')