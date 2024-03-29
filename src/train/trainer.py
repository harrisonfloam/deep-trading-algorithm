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
from src.train.utils import EarlyStopping, TensorBoardLogger, TrainingState, auto_close_tensorboard, plot_data, plot_learning_curves, save_model, load_model
from src.utils import print_to_console, update_progress


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
        
        # Logging
        self.tensorboard_logger = TensorBoardLogger(self.use_tensorboard, verbose=self.verbose)
        
    def __del__(self):
        self.tensorboard_logger.stop()

    @auto_close_tensorboard
    def train(self, train_loader, val_loader, epochs, lr, no_change_patience, overfit_patience, warmup, save_best=False):
        # Training Parameters
        self.epochs = epochs
        self.lr = lr
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')
        self.early_stopper = EarlyStopping(no_change_patience=no_change_patience,
                                           overfit_patience=overfit_patience,
                                           warmup=warmup, verbose=self.verbose)
        self.state = TrainingState()

        # Logging
        print_to_console(verbose=self.verbose, mode='train', model_name=self.model_name)
        tqdm_epochs = tqdm(range(epochs), disable=not self.verbose, desc='Epochs')
        
        # Training loop
        for epoch in tqdm_epochs:
            self.model.train()
            batch_losses = []
            start_time = time.time()

            for i, (x, y) in enumerate(train_loader):
                update_progress(tqdm_instance=tqdm_epochs, mode='train', section='desc', content=[i, train_loader])
                
                x, y = x.to(self.device), y.to(self.device)

                # Forward pass
                outputs = self.model(x).squeeze()  # Remove extra dimension from model's output TODO: see if i should just do this in forward
                loss = self.criterion(outputs, y)

                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_losses.append(loss.item())
                
                update_progress(tqdm_instance=tqdm_epochs, mode='train', section='postfix',
                                content=[loss, batch_losses, self.state])
            
            # Training metrics
            elapsed_time = time.time() - start_time
            mean_loss_epoch = np.mean(batch_losses)
            self.state.update(epoch=epoch, elapsed_time=elapsed_time, mean_loss_epoch=mean_loss_epoch)   # Update state
            
            # Validation loop
            val_loss = self.evaluate(val_loader, show_progress=False)
            self.scheduler.step(val_loss)    # Learning rate scheduler step
            self.state.update(val_loss=val_loss)    # Update state
            
            # Tensorboard logging
            self.tensorboard_logger.log_loss(self.state)
            self.tensorboard_logger.log_params_grads(model=self.model, epoch=epoch)
            
            # Early stopping and model saving
            stop, new_best = self.early_stopper.should_stop(self.state)
            if new_best:
                if save_best:
                    save_model(model=self.model, filename=self.model_name)            
            if stop:
                break

    def evaluate(self, loader, show_progress=True):
        print_to_console(mode='val', model_name=self.model_name, verbose=self.verbose, show_progress=show_progress)
        tqdm_batches = tqdm(loader, disable=not (self.verbose and show_progress))

        self.model.eval()  # Set model to evaluation mode
        total_loss = 0.0
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm_batches):
                x, y = x.to(self.device), y.to(self.device) # Move tensors to device
                outputs = self.model(x).squeeze()  # Forward pass
                loss = self.criterion(outputs, y)  # Compute loss
                total_loss += loss.item()
                mean_loss_epoch = total_loss/(i + 1)
                
                update_progress(tqdm_instance=tqdm_batches, mode='val', section='postfix',
                                content=[loss, mean_loss_epoch])

        mean_loss = total_loss / len(loader)

        return mean_loss