# Time series LSTM optimization
# Harrison Floam, 26 July 2023

# Import Libraries
import numpy as np
import pandas as pd
import datetime as dt
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from torchinfo import summary
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler
from ray import tune, air
from ray.air import session
from ray.tune.search.optuna import OptunaSearch


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len, num_layers):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.input_dim = input_dim

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.fc1 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),  # New fully connected layer
            nn.ReLU()  # Activation function
        )
        self.fc2 = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])  # Pass through new fully connected layer
        out = self.fc2(out)
        return out

    # torch.save(model.state_dict(), PATH)
    # model.load_state_dict(torch.load(PATH))

class DataSetLoader:
    def __init__(self, path="data/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv", start='2021-03-27 15:04:00', end='2021-03-31 00:00:00', verbose=False):
        self.path = path    # Filepath of data
        self.start = start
        self.end = end
        self.verbose = verbose

    def load_data(self):
        # Import Data
        self.data = pd.read_csv(self.path)

    def clean_data(self):
        # Clean data
        self.data.fillna(method="ffill", inplace=True)  # Forward/backward fill prices
        self.data.fillna(method="bfill", inplace=True)

    def add_features(self):
        self.data['SMA'] = ta.sma(close=self.data['Close'], length=20)    # Simple Moving Average
        self.data['RSI'] = ta.rsi(close=self.data['Close'], length=14)    # Relative Strength Index
        macd = ta.macd(close=self.data['Close'], fast=12, slow=26, signal=9)  # MACD
        self.data['MACD'] = macd['MACD_12_26_9']
        #self.data['bb_upper'], self.data['bb_middle'], self.data['bb_lower'] = ta.bbands(close=self.data['Close'], length=20) # Bollinger Bands
        #self.data['VWAP'] = ta.vwap(high=self.data['High'], low=self.data['Low'], close=self.data['Close'], volume=self.data['Volume_(BTC)'])

        self.data.dropna(inplace=True)  # Drop rows with NaN values
        # self.data.reset_index(drop=True, inplace=True)  # Reset the index

    def trim_data(self):
        # Convert to datetime index
        self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'], unit='s') # Convert the Timestamp column to datetime
        self.data.set_index('Timestamp', inplace=True)

        # Convert start and end to datetime
        start = pd.to_datetime(self.start)
        end = pd.to_datetime(self.end)

        # Select rows between start and end dates
        self.data = self.data.loc[start:end]

        if self.verbose:
            print(f'Data loaded --------- {self.path}')
            print(f'Using data from {start} to {end} with {len(self.data)} rows')

    def split_data(self):
        # Split data
        self.data.sort_values('Timestamp', inplace=True) # Sort the dataframe by timestamp
        train_val, self.test = train_test_split(self.data, test_size=0.15, shuffle=False) # Split the data
        self.train, self.val = train_test_split(train_val, test_size=0.176, shuffle=False)  # Create validation set from train

    def scale_data(self):
        # Initialize dictionary to store individual scalers for each feature
        self.scaler_features = {col: MinMaxScaler(feature_range=(-1, 1)) for col in self.train.columns if col != 'Close'}

        # Initialize scaler for the 'Close' column
        self.scaler_close = MinMaxScaler(feature_range=(-1, 1))

        # Scale 'Close' column
        self.train['Close'] = self.scaler_close.fit_transform(self.train['Close'].values.reshape(-1, 1))
        self.val['Close'] = self.scaler_close.transform(self.val['Close'].values.reshape(-1, 1))
        self.test['Close'] = self.scaler_close.transform(self.test['Close'].values.reshape(-1, 1))

        # Scale all other features individually
        for col in self.scaler_features.keys():
            self.train[col] = self.scaler_features[col].fit_transform(self.train[col].values.reshape(-1, 1))
            self.val[col] = self.scaler_features[col].transform(self.val[col].values.reshape(-1, 1))
            self.test[col] = self.scaler_features[col].transform(self.test[col].values.reshape(-1, 1))

        if self.verbose: self.test.describe()

    def unscale_data(self, df, scaler):
        df_index = df.index     # Store index and columns
        df_columns = df.columns

        df_unscaled = np.squeeze(scaler.inverse_transform(df.values.reshape(-1,1))) # Inverse transform
        df = pd.DataFrame(df_unscaled, columns=df_columns, index=df_index)  # Re-create dataframes

        return df

    def prepare_data(self):
        self.load_data()
        self.clean_data()
        self.add_features()
        self.trim_data()
        self.split_data()
        self.scale_data()

class DataFormatter:
    def __init__(self, df, window, batch_size):
        self.df = df
        self.window = window
        self.batch_size = batch_size

    class TimeSeriesDataset(Dataset):
        def __init__(self, df, window):
            self.df = df
            self.window = window

        def __len__(self):
            return len(self.df) - self.window

        def __getitem__(self, idx):
            x = self.df.iloc[idx : idx + self.window].drop(columns='Close').values
            y = self.df.iloc[idx + self.window]['Close']  # get 'Close' price at the next timestep

            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def create_dataloader(self):
        dataset = self.TimeSeriesDataset(df=self.df, window=self.window)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return dataloader

class Trainer:
    def __init__(self,
                 data_percent=0.001,
                 path="data/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv",
                 start='2021-03-27 00:00:00',
                 end='2021-03-31 00:00:00',
                 window = 10,
                 batch_size = 256,
                 hidden_dim = 128,
                 num_layers = 1,
                 output_dim = 1,
                 verbose=False,
                 runningInColab=False,
                 use_TPU=False):
        # Data parameters
        self.path = path
        self.start = start
        self.end = end

        # Model parameters
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.window = window
        self.num_layers = num_layers
        self.batch_size = batch_size

        # Debug
        self.verbose = verbose
        self.runningInColab = runningInColab
        self.use_TPU = use_TPU

    def get_data(self):
        self.dataset = DataSetLoader(path=self.path, start=self.start, end=self.end, verbose=self.verbose)
        self.dataset.prepare_data()
        self.input_dim = len(self.dataset.data.columns) - 1   # drop close

    def initialize_model(self):
        self.model = LSTM(self.input_dim, self.hidden_dim, self.output_dim, self.window, self.num_layers)
        self.model_name = self.model.__class__.__name__

    def create_dataloaders(self):
        train_formatter = DataFormatter(df=self.dataset.train, window=self.window, batch_size=self.batch_size)
        val_formatter = DataFormatter(df=self.dataset.val, window=self.window, batch_size=self.batch_size)
        test_formatter = DataFormatter(df=self.dataset.test, window=self.window, batch_size=self.batch_size)

        self.train_loader = train_formatter.create_dataloader()
        self.val_loader = val_formatter.create_dataloader()
        self.test_loader = test_formatter.create_dataloader()

    def setup_model(self):
        self.get_data()
        self.initialize_model()
        self.create_dataloaders()

    def train_model(self, device, epochs=10, lr=0.001, min_loss=0.01, patience=5, loss_std_tol=0.0003):
        # Training Parameters
        self.epochs = epochs
        self.lr = lr
        self.min_loss = min_loss
        self.patience = patience
        self.loss_std_tol = loss_std_tol
        self.device = device
        self.criterion = nn.MSELoss()  # Mean Squared Error Loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model = self.model.to(self.device)

        if self.use_TPU:
            train_loader = pl.MpDeviceLoader(self.train_loader, xm.xla_device())
        else:
            train_loader = self.train_loader

        if self.verbose:
            print(f'Training --------- Model: {self.model_name}')

        train_elapsed_time = 0
        recent_losses = []

        for epoch in range(epochs):
            self.model.train()
            train_losses = []
            start_time = time.time()

            progress_stars = '' # Progress star tracking

            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)

                # Forward pass
                outputs = self.model(x).squeeze()  # Remove extra dimension from model's output
                loss = self.criterion(outputs, y)

                progress_stars = self.progress_stars(progress_stars, i, train_loader) # Print progress

                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()

                progress_stars = self.progress_stars(progress_stars, i, train_loader) # Print progress

                if self.use_TPU:
                    xm.optimizer_step(optimizer=self.optimizer)
                    xm.mark_step()
                    # if epoch == 0:
                    #     print('Tensor device:', x.get_device())
                else:
                    self.optimizer.step()

                train_losses.append(loss.item())

                progress_stars = self.progress_stars(progress_stars, i, train_loader) # Print progress

            elapsed_time = time.time() - start_time
            train_elapsed_time += elapsed_time

            mean_loss = np.mean(train_losses)
            recent_losses.append(mean_loss) # Recent losses list
            if len(recent_losses) > self.patience:
                recent_losses.pop(0)

            if epoch == 0:
                original_loss = mean_loss
            loss_decrease = mean_loss/original_loss # Loss decrease percent from original

            loss_std = np.std(recent_losses)    # Std of recent losses

            if self.verbose:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {mean_loss:.4f}, Loss decrease: {loss_decrease:.5f}/{self.min_loss}, Loss std: {loss_std:.5f}/{self.loss_std_tol}, Time Elapsed: {elapsed_time:.2f}s')

            xla_debug = True
            if self.use_TPU:
                if xla_debug:
                    print('Model device:', next(self.model.parameters()).device)
                    print(met.short_metrics_report())

            # Termination conditions TODO: stop when validation loss starts decreasing
            # Stop training if loss decreases signficantly
            if loss_decrease < self.min_loss:
                break
            # Stop training if loss flatlines
            if len(recent_losses) == self.patience and loss_std < self.loss_std_tol:
                break

        if self.verbose:
            print(f'Total Train Time: {train_elapsed_time:.2f}s')

    def evaluate_model(self, loader, set_name):
        self.model.eval()  # Set model to evaluation mode
        start_time = time.time()

        if self.verbose:
            print(f'Evaluating {set_name} Set --------- Model: {self.model_name}')
        
        progress_stars = ''

        total_loss = 0.0
        with torch.no_grad():  # No need to track gradients
            for i, (x, y) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)

                progress_stars = self.progress_stars(progress_stars, i, loader) # Print progress

                outputs = self.model(x).squeeze()  # Forward pass

                progress_stars = self.progress_stars(progress_stars, i, loader) # Print progress

                loss = self.criterion(outputs, y)  # Compute loss

                progress_stars = self.progress_stars(progress_stars, i, loader) # Print progress

                total_loss += loss.item()
        elapsed_time = time.time() - start_time

        mean_loss = total_loss / len(loader)

        if self.verbose:
            print(f'Loss: {mean_loss:.4f}, Time Elapsed: {elapsed_time:.2f}s')

        return mean_loss

    def predict(self, loader, set_name):
        self.model.eval()  # Set model to evaluation mode
        start_time = time.time()

        if self.verbose:
            print(f'Predicting {set_name} Set --------- Model: {self.model_name}')

        progress_stars = ''

        predictions = []
        timestamps = []
        with torch.no_grad():  # No need to track gradients
            for i, (x, _) in enumerate(loader):
                x = x.to(self.device)

                progress_stars = self.progress_stars(progress_stars, i, loader) # Print progress

                outputs = self.model(x)  # Forward pass

                progress_stars = self.progress_stars(progress_stars, i, loader) # Print progress

                scaler_debug = False
                if scaler_debug:
                    print(f'\tBatch {i+1}, Scaler range: {loader.dataset.scaler_close.data_range_}, Scaled rows: {loader.dataset.scaler_close.n_samples_seen_}')

                # Flatten the list and extend the predictions
                predictions.extend([item for sublist in outputs.tolist() for item in sublist])  # Flatten the list
                batch_timestamps = loader.dataset.df.index[i*loader.batch_size + self.window : i*loader.batch_size + self.window + len(x)].tolist()
                timestamps.extend(batch_timestamps)
                
                progress_stars = self.progress_stars(progress_stars, i, loader) # Print progress

        elapsed_time = time.time() - start_time

        if self.verbose:
            print(f'Time Elapsed: {elapsed_time:.2f}s')

        return pd.DataFrame({'Timestamp': timestamps, self.model_name: predictions}).set_index('Timestamp')

    def plot_data(self, actual, predicted, set_name, loss, unscale=True):
        # Unscale data
        if unscale:
            actual = self.dataset.unscale_data(actual[['Close']], self.dataset.scaler_close)
            predicted = self.dataset.unscale_data(predicted, self.dataset.scaler_close)
            unit = 'USD'
        else:
            actual = actual[['Close']]
            unit = 'Normalized'

        if self.verbose:
            actual.describe()
            predicted.describe()

        sns.set_style('darkgrid')
        plt.figure(figsize=(10,6))

        plt.plot(actual.index, actual['Close'], label='Actual', color='green', lw=1)
        plt.plot(predicted.index, predicted[self.model_name], label='Predicted', color='red', lw=1)

        # Add vertical lines for the window
        for idx in range(self.window, len(actual), self.window):
            plt.axvline(x=actual.index[idx], color='orange', linestyle='--', lw=0.5)


        plt.title(f'BTC Close Over Time ({set_name}, MSE={loss:.5f})', fontsize=16)  # Set the title
        plt.xlabel('Time', fontsize=14)  # Set x-axis label
        plt.ylabel(f'Close Price ({unit})', fontsize=14)  # Set y-axis label

        plt.legend()  # Show the legend
        plt.show(block=False)  # Show the plot

    def progress_stars(self, progress_stars, idx, loader):
        num_batches = int(len(loader))
        batch_number = f'{idx + 1}/{num_batches}'
        max_chars = 189
        max_chars_padded = max_chars - len(f'{batch_number}')*2 - 1

        if num_batches*3 > max_chars_padded:
            idx_star_batch = int(num_batches / (max_chars_padded / 3)) + 1
            if idx % idx_star_batch == 0:
                if self.verbose:
                    progress_stars += '*'
                    print(f'\r{progress_stars} {batch_number}', end='')
        else:
            progress_stars += '*'
            print(f'\r{progress_stars} {batch_number}', end='')
        if idx + 1 == num_batches:
            if self.verbose:
                print(f'\r{" " * max_chars}\r', end='')
        return progress_stars

def objective(config):
    # Initialize Trainer
    trainer = Trainer(
        data_percent=0.001,
        path="home/time-series-project/data/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv",
        window=config["window"],
        batch_size=config["batch_size"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        output_dim=1,
        verbose=True
    )
    trainer.setup_model()
    model = trainer.model

    # Initialize optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])

    # Define criterion
    criterion = nn.MSELoss()

    # Train and evaluate model
    trainer.train_model()
    val_loss = trainer.evaluate_model(trainer.val_loader)

    # Report to Tune
    session.report({"val_loss": val_loss})

def optimize():
    # Optuna optimization
    search_space = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "momentum": tune.uniform(0.1, 0.9),
        "window": tune.randint(5, 20),  # integer from 5 to 20
        "batch_size": tune.choice([32, 64, 128, 256, 512]),  # choose from list of options
        "hidden_dim": tune.randint(64, 256),  # integer from 64 to 256
        "num_layers": tune.randint(1, 3),  # integer from 1 to 3
    }
    algo = OptunaSearch()
    tuner = tune.Tuner(objective, tune_config=tune.TuneConfig(metric="val_loss", mode="min", search_alg=algo),
    run_config=air.RunConfig(stop={"training_iteration": 5}), param_space=search_space)
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)

def test(runningInColab, use_TPU, path, device):
    trainer = Trainer(path=path+'bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv',
                      start='2021-03-20 00:00:00',
                      end='2021-03-25 00:00:00',
                      window=60,
                      batch_size=256,
                      hidden_dim=128,#512, #128
                      num_layers=1,
                      output_dim=1,
                      verbose=True,
                      runningInColab=runningInColab,
                      use_TPU=use_TPU)
    trainer.setup_model()
    trainer.train_model(epochs=100,
                        lr=0.001,
                        min_loss=0.001,
                        patience=5,
                        loss_std_tol=0.00017,
                        device=device)

    # Evaluate training set
    train_loss = trainer.evaluate_model(trainer.train_loader, set_name='Train')
    train_actual = trainer.dataset.train
    train_predicted = trainer.predict(trainer.train_loader, set_name='Train')
    trainer.plot_data(train_actual, train_predicted, set_name='Train', loss=train_loss, unscale=True)

    # Evaluate validation set
    val_loss = trainer.evaluate_model(trainer.val_loader, set_name='Validation')
    val_actual = trainer.dataset.val
    val_predicted = trainer.predict(trainer.val_loader, set_name='Validation')
    trainer.plot_data(val_actual, val_predicted, set_name='Validation', loss=val_loss, unscale=True)

    # Evaluate test set
    test_loss = trainer.evaluate_model(trainer.test_loader, set_name='Test')
    test_actual = trainer.dataset.test
    test_predicted = trainer.predict(trainer.test_loader, set_name='Test')
    trainer.plot_data(test_actual, test_predicted, set_name='Test', loss=test_loss, unscale=True)

    plt.show()

def google_colab_handler(use_TPU):
    try:
        from google.colab import drive
        runningInColab = True
    except:
        runningInColab = False

    # runningInColab = 'google.colab' in str(get_ipython())

    if runningInColab:
        print('Running in Google Colab')
        path = '/content/drive/MyDrive/Google Colab/time-series-project/'
        if use_TPU:
            import torch_xla
            import torch_xla.core.xla_model as xm
            import torch_xla.debug.metrics as met
            import torch_xla.distributed.parallel_loader as pl

            device = xm.xla_device()
            if xm.xla_device().type == 'xla':
                print('TPU is available')
            # Initialize the TPU cores
            xm.rendezvous('init')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available, else CPU
    else:
        path = 'working/data/'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available, else CPU
        if device == 'cuda':
            torch.cuda.empty_cache()
    print(f"Using Device: {device}")

    return runningInColab, use_TPU, path, device


if __name__ == '__main__':
    runningInColab, use_TPU, path, device = google_colab_handler(use_TPU=False)
    test(runningInColab, use_TPU, path, device)
    # optimize()
