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
from src.utils import print_to_console, update_progress

from train.utils import EarlyStopping, TensorBoardLogger


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
        
        # Logging
        self.tensorboard_logger = TensorBoardLogger(self.use_tensorboard)
        
    def __del__(self):
        self.tensorboard_logger.stop()

    def train_model(self, train_loader, val_loader, epochs, lr, no_change_patience, overfit_patience, warmup, save_best=False):
        # Training Parameters
        self.epochs = epochs
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(self.optimizer, 'min')
        self.model = self.model.to(self.device)
        self.early_stopper = EarlyStopping(no_change_patience=no_change_patience,
                                           overfit_patience=overfit_patience,
                                           warmup=warmup)

        self.train_loader = train_loader
        self.val_loader = val_loader

        print_to_console(verbose=self.verbose, mode='train', model_name=self.model_name)
            
        if self.use_tensorboard:
            self.tensorboard_process = self.start_tensorboard()

        train_losses = []
        val_losses = []
        
        train_elapsed_time = 0
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
                outputs = self.model(x).squeeze()  # Remove extra dimension from model's output TODO: see if i should just do this in forward
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
            val_loss = self.evaluate_model(val_loader, show_progress=False)
            val_losses.append(val_loss)
            mean_loss_val = np.mean(val_losses)
            scheduler.step(val_loss)    # Learning rate scheduler step
            
            # Tensorboard logging
            self.tensorboard_logger.log_loss(train_loss=mean_loss_epoch, val_loss=val_loss, epoch=epoch)
            self.tensorboard_logger.log_params_grads(model=self.model, epoch=epoch)
            
            # Early stopping
            if self.early_stopper.should_stop(val_loss, mean_loss_val, mean_loss_epoch, epoch):
                break
            
        # Save learning data
        self.train_losses = train_losses
        self.val_losses = val_losses
        #TODO: save model...

    def evaluate_model(self, loader, show_progress=True):
        # Logging
        print_to_console(mode='val', model_name=self.model_name, verbose=self.verbose, show_progress=show_progress)
        tqdm_batches = tqdm(loader, disable=not (self.verbose and show_progress))

        self.model.eval()  # Set model to evaluation mode
        total_loss = 0.0
        start_time = time.time()
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm_batches):
                x, y = x.to(self.device), y.to(self.device) # Move tensors to device
                outputs = self.model(x).squeeze()  # Forward pass
                loss = self.criterion(outputs, y)  # Compute loss
                total_loss += loss.item()
                mean_loss_epoch = total_loss/(i + 1)
                
                update_progress(tqdm_instance=tqdm_batches, mode='val', section='postfix',
                                content=[loss, mean_loss_epoch])
            
        elapsed_time = time.time() - start_time

        mean_loss = total_loss / len(loader)

        return mean_loss