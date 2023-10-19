"""
Data preprocessing and loading utility functions
"""

# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Import Modules
# from src.data.data import DataBundle


class DataBundle:
    def __init__(self, data=None, train=None, test=None, val=None, scaler=None):
        self.data = data
        self.train = train
        self.test = test
        self.val = val
        self.scaler = scaler
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)
                
    def get_bundle(self):
        return self.train, self.test, self.val

def to_datetime_index(df, index_col):
    """Converts a dataframe's index into a datetime index
    """
    df[index_col] = pd.to_datetime(df[index_col], unit='s') # Convert the index_col column to datetime
    df.set_index(index_col, inplace=True)
    return df

def forward_backward_fill(df):
    """Forward, then backward fills NA's in a dataframe
    """
    df.fillna(method="ffill", inplace=True)   # Forward/backward fill prices
    df.fillna(method="bfill", inplace=True)
    return df

def train_test_val_split(input, test_percent, val_percent):
    """Splits a dataframe or databundle into train, test, and val sets based on input percents
    """
    if isinstance(input, DataBundle):
        df = input.data
    else:
        df = input
    val_split_percent = val_percent/(1 - test_percent)    
    train_val, test = train_test_split(df, test_size=test_percent, shuffle=False) # Split the data
    train, val = train_test_split(train_val, test_size=val_split_percent, shuffle=False)  # Create validation set from train
    if isinstance(input, DataBundle):
        input.update(train=train, test=test, val=val)
    else:
        return train, test, val

def scale_data(databundle, exclude_columns): 
    """Scales train, test, and val dataframes held in a databundle. Scaler is fit to train set.
    """
    train, test, val = databundle.get_bundle()
    for col in train.columns:
        if col in exclude_columns:
            continue
        if col == 'percent_ret':
            percent_ret_scaler = MinMaxScaler(feature_range=(0, 1)) #StandardScaler()
            scaler = percent_ret_scaler
        else:
            scaler = MinMaxScaler(feature_range=(0, 1))
            percent_ret_scaler = None
            
        train[col] = scaler.fit_transform(train[col].values.reshape(-1, 1))
        val[col] = scaler.transform(val[col].values.reshape(-1, 1))
        test[col] = scaler.transform(test[col].values.reshape(-1, 1))
        
    databundle.update(train=train, test=test, val=val, scaler=percent_ret_scaler)
    
def unscale_data(scaled_df, databundle):
    if databundle.scaler is None:
        raise ValueError("The target column has not been scaled yet.")
    
    scaled_percent_ret = scaled_df['percent_ret'].values
    unscaled_percent_ret = databundle.scaler.inverse_transform(scaled_percent_ret.reshape(-1, 1))
    
    unscaled_df = scaled_df.copy()
    unscaled_df['percent_ret'] = unscaled_percent_ret.flatten()
    return unscaled_df
    
