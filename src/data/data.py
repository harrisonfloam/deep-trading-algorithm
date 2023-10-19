"""
Data processing and loading
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

# Import modules
from src.utils import get_project_root
from src.data.utils import forward_backward_fill, to_datetime_index, train_test_val_split, scale_data, unscale_data, DataBundle


class DataProcessor:
    def __init__(self, start, end, verbose=False):
        self.start = pd.to_datetime(start)
        self.end = pd.to_datetime(end)
        self.verbose = verbose

    def load_data(self, filename):
        self.path = os.path.join(get_project_root(), 'data', filename)    # Filepath of data
        data = pd.read_csv(self.path)
        databundle = DataBundle(data=data)
        if self.verbose:
            print(f'Data loaded --------- data/{filename}')
        return databundle

    def clean_data(self, databundle):
        data = to_datetime_index(databundle.data, 'Timestamp') # Convert to datetime index
        data = forward_backward_fill(data)
        data.sort_values('Timestamp', inplace=True) # Sort the dataframe by timestamp
        databundle.update(data=data)    # Update data

    def add_features(self, databundle):
        data = databundle.data  # Extract data from bundle
        data['SMA'] = ta.sma(close=data['Close'], length=20)    # Simple Moving Average
        data['RSI'] = ta.rsi(close=data['Close'], length=14)    # Relative Strength Index
        macd = ta.macd(close=data['Close'], fast=12, slow=26, signal=9)  # MACD
        data['MACD'] = macd['MACD_12_26_9']
        #self.data['bb_upper'], self.data['bb_middle'], self.data['bb_lower'] = ta.bbands(close=self.data['Close'], length=20) # Bollinger Bands
        #self.data['VWAP'] = ta.vwap(high=self.data['High'], low=self.data['Low'], close=self.data['Close'], volume=self.data['Volume_(BTC)'])
        data['log_ret'] = ta.log_return(close=data['Close'], cumulative=False)    # Log returns
        data['percent_ret'] = ta.percent_return(close=data['Close'], cumulative=False)*10    # Percent returns
        data.dropna(inplace=True)  # Drop rows with NaN values
        
        databundle.update(data=data)    # Update data

    def trim_data(self, databundle):
        data = databundle.data  # Extract data from bundle
        data = data.loc[self.start:self.end]    # Select rows between start and end dates
        if self.verbose:
            print(f'Using data from {self.start} to {self.end} with {len(data)-1} rows and {len(data.columns)} columns')
        databundle.update(data=data)    # Update data

    def split_data(self, databundle):
        train_test_val_split(input=databundle, test_percent=0.15, val_percent=0.15)

    def scale_data(self, databundle, exclude_columns): 
        scale_data(databundle, exclude_columns)

    def unscale_data(self, scaled_df, databundle):
        #TODO: make sure we can access databundle later...
        unscaled_df = unscale_data(scaled_df=scaled_df, databundle=databundle)
        return unscaled_df
    
    def normalize_with_rolling_stats(self, window_size):
        for col in self.data.columns:
            rolling_mean = self.data[col].rolling(window=window_size).mean()
            rolling_std = self.data[col].rolling(window=window_size).std()
            self.data[col] = (self.data[col] - rolling_mean) / rolling_std
        self.data.dropna(inplace=True)
        
    def detrend_data(self, databundle, exclude_columns):
        train, test, val = databundle.get_bundle()
        for col in train.columns:
            if col in exclude_columns:
                continue
            linreg = LinearRegression()
            for subset in [train, test, val]:
                x = np.arange(len(subset)).reshape(-1, 1)
                y = subset[col].values
                if subset is train:
                    linreg.fit(x, y)
                trend = linreg.predict(x)
                subset[col] = subset[col].values - trend
        databundle.update(train=train, test=test, val=val)

    def prepare_data(self, filename, exclude_columns_detrend, exclude_columns_scale):
        databundle = self.load_data(filename)
        self.clean_data(databundle)
        self.add_features(databundle)
        self.trim_data(databundle)
        self.split_data(databundle)
        self.detrend_data(databundle=databundle, exclude_columns=exclude_columns_detrend)
        self.scale_data(databundle=databundle, exclude_columns=exclude_columns_scale)
        self.databundle = databundle
        
    def create_dataloaders(self, window, batch_size, exclude_input_columns):
        train, test, val = self.databundle.get_bundle()
        train_loader = TimeSeriesDataLoader(df=train, window=window, batch_size=batch_size, 
                                            exclude_input_columns=exclude_input_columns)
        test_loader = TimeSeriesDataLoader(df=test, window=window, batch_size=batch_size, 
                                            exclude_input_columns=exclude_input_columns)
        val_loader = TimeSeriesDataLoader(df=val, window=window, batch_size=batch_size, 
                                            exclude_input_columns=exclude_input_columns)
        return train_loader, test_loader, val_loader

    
class TimeSeriesDataset(Dataset):
    def __init__(self, df, window, exclude_input_columns):
        self.window = window
        self.exclude_input_columns = [col for col in exclude_input_columns if col in df.columns]
        self.input_df = df.drop(columns=self.exclude_input_columns)
        self.target_series = df['percent_ret']
        
    def __len__(self):
        return len(self.input_df) - self.window

    def __getitem__(self, idx):
        x = self.input_df.iloc[idx : idx + self.window].values
        y = self.target_series.iloc[idx + self.window]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    
class TimeSeriesDataLoader(DataLoader):  # Inherits from DataLoader
    def __init__(self, df, window, batch_size, exclude_input_columns=['']):
        dataset = TimeSeriesDataset(df=df, window=window, exclude_input_columns=exclude_input_columns)
        super(TimeSeriesDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=False)
