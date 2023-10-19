"""
Trainer class definition
"""

# Import Libraries
import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchinfo import summary
from tqdm.autonotebook import tqdm

# Import Modules
from src.train.utils import plot_data, plot_learning_curves, start_tensorboard, stop_tensorboard, save_model, load_model
from src.utils import update_progress


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
                update_progress(tqdm_instance=tqdm_epochs, mode='train', section='desc', content=[i, train_loader])
                
                x, y = x.to(self.device), y.to(self.device)

                # Forward pass
                outputs = self.model(x).squeeze()  # Remove extra dimension from model's output
                loss = self.criterion(outputs, y)

                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_losses.append(loss.item())
                
                update_progress(tqdm_instance=tqdm_epochs, mode='train', section='postfix',
                                content=[loss, batch_losses, mean_loss_epoch, mean_loss_training, val_loss, mean_loss_val])

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
                self.writer.add_scalar('Training Loss', mean_loss_epoch, epoch) #BUG
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
                    self.save_model(model_path=f"saved_models/{self.model_name}.pth")   #BUG
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