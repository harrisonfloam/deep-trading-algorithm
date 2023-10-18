"""
Data Processing and Loading for deep-trading-algorithim
"""

# Import Libraries
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression


class DataProcessor:
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

        self.data['log_ret'] = ta.log_return(close=self.data['Close'], cumulative=False)    # Log returns
        self.data['percent_ret'] = ta.percent_return(close=self.data['Close'], cumulative=False)*10    # Percent returns

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
            print(f'Using data from {start} to {end} with {len(self.data)-1} rows')

    def split_data(self):
        # Split data
        self.data.sort_values('Timestamp', inplace=True) # Sort the dataframe by timestamp
        train_val, self.test = train_test_split(self.data, test_size=0.15, shuffle=False) # Split the data
        self.train, self.val = train_test_split(train_val, test_size=0.176, shuffle=False)  # Create validation set from train

    def scale_data(self, exclude_columns): 
        for col in self.data.columns:
            if col in exclude_columns:
                continue
            if col == 'percent_ret':
                self.percent_ret_scaler = StandardScaler()
                scaler = self.percent_ret_scaler
            else:
                scaler = MinMaxScaler(feature_range=(0, 1))
            self.train[col] = scaler.fit_transform(self.train[col].values.reshape(-1, 1))
            self.val[col] = scaler.transform(self.val[col].values.reshape(-1, 1))
            self.test[col] = scaler.transform(self.test[col].values.reshape(-1, 1))

    def unscale_data(self, scaled_df):
        if self.percent_ret_scaler is None:
            raise ValueError("The 'percent_ret' column has not been scaled yet.")
        
        scaled_percent_ret = scaled_df['percent_ret'].values
        unscaled_percent_ret = self.percent_ret_scaler.inverse_transform(scaled_percent_ret.reshape(-1, 1))
        
        unscaled_df = scaled_df.copy()
        unscaled_df['percent_ret'] = unscaled_percent_ret.flatten()
        return unscaled_df

    
    def normalize_with_rolling_stats(self, window_size):
        for col in self.data.columns:
            rolling_mean = self.data[col].rolling(window=window_size).mean()
            rolling_std = self.data[col].rolling(window=window_size).std()
            self.data[col] = (self.data[col] - rolling_mean) / rolling_std
        self.data.dropna(inplace=True)
        
    def detrend_data(self, exclude_columns):
        for col in self.data.columns:
            if col in exclude_columns:
                continue
            linreg = LinearRegression()
            x = np.arange(len(self.data)).reshape(-1, 1)
            y = self.data[col].values
            linreg.fit(x, y)
            trend = linreg.predict(x)
            self.data[col] = y - trend

    def prepare_data(self, exclude_columns_detrend, exclude_columns_scale):
        self.load_data()
        self.clean_data()
        self.add_features()
        self.trim_data()
        self.split_data()
        self.detrend_data(exclude_columns=exclude_columns_detrend)
        self.scale_data(exclude_columns=exclude_columns_scale)
        
    def create_dataloaders(self, window, batch_size, exclude_input_columns):
        train_loader = TimeSeriesDataLoader(df=self.train, window=window, 
                                            batch_size=batch_size, 
                                            exclude_input_columns=exclude_input_columns).create_dataloader()
        val_loader = TimeSeriesDataLoader(df=self.val, window=window, 
                                          batch_size=batch_size, 
                                          exclude_input_columns=exclude_input_columns).create_dataloader()
        test_loader = TimeSeriesDataLoader(df=self.test, window=window, 
                                           batch_size=batch_size, 
                                           exclude_input_columns=exclude_input_columns).create_dataloader()
        return train_loader, val_loader, test_loader

class TimeSeriesDataset(Dataset):
    def __init__(self, df, window, exclude_input_columns):
        self.df = df
        self.window = window
        self.exclude_input_columns = exclude_input_columns

    def __len__(self):
        return len(self.df) - self.window

    def __getitem__(self, idx):
        drop_columns = [col for col in self.exclude_input_columns if col in self.df.columns]
        x_df = self.df.iloc[idx : idx + self.window].drop(columns=drop_columns)
        x = x_df.values
        y = self.df.iloc[idx + self.window]['percent_ret']  # Percent return at the next timestep
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
class TimeSeriesDataLoader:
    def __init__(self, df, window, batch_size, exclude_input_columns=['']):
        self.df = df
        self.window = window
        self.batch_size = batch_size
        self.exclude_input_columns = exclude_input_columns
        self.dataset = TimeSeriesDataset(df=self.df, window=self.window, exclude_input_columns=self.exclude_input_columns)

    def create_dataloader(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        return dataloader