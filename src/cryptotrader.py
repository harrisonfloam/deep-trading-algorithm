## Crypto Trading Algorithm
# Harrison Floam, 10 April 2023

# Import Libraries
import time
import cbpro
import datetime
import pandas as pd
import pandas_ta as ta
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Import Modules
from .cryptomodel import CryptoModel
from .coinbase import CoinbaseAPI


class CryptoTrader:
    """
    A class for creating and training a LSTM-based cryptocurrency trading algorithm. 

    ### Usage:
    -----------
    Initialize, then call train(), then run(), or train_run().

    ### Methods:
    -----------
    - train(self, data, batch_size=32, epochs=10)
        Trains the LSTM model on the given historical price data.
    - run()
        Starts the live trading loop.
    - train_run()
        Train the model and start the live trading loop in one call.
    - test_train()
        Train the model with mock data
    - test_run()
        Run the live trading loop with mock data
    """
    def __init__(self, initial_balance, trade_interval, run_time, model_class, 
                 run_time_unit='h', product_id='BTC', buy_threshold=0.02, sell_threshold=0.02,
                 order_p=0.1, confidence_threshold=0.80, slippage_p=0.005, fees_p=0.005, indicators=True, verbose=False):
        # Initialize trading parameters
        self.product_id = product_id                        # Crypto symbol
        self.initial_balance = initial_balance              # Starting balance to trade
        self.trade_interval = trade_interval                # Interval to place trades (s)
        self.buy_threshold = buy_threshold                  # Buy threshold percentage
        self.sell_threshold = sell_threshold                # Sell threshold percentage
        self.order_p = order_p                              # Percent of wallet balance to trade
        self.confidence_threshold = confidence_threshold    # Prediction confidence at which to trade
        self.slippage_p = slippage_p                        # Slippage percentage
        self.fees_p = fees_p                                # Fee percentage

        # Initialize data and model parameters
        self.model_class = model_class      # Model class
        self.price_data = pd.DataFrame()    # Price dataframe
        self.model = None                   # LSTM model
        self.criterion = None               # Model criterion
        self.optimizer = None               # Model optimizer
        self.hidden = None                  # Hidden layers
        self.indicators = indicators        # Indicator toggle

        # Initialize Coinbase API parameters
        self.coinbase_api = CoinbaseAPI(product_id=product_id)

        # Initialize time parameters
        self.run_time = run_time
        self.run_time_unit = run_time_unit

        # Initialize tracking parameters
        self.order_log = pd.DataFrame()
        self.verbose = verbose
        self.test = False

    # Initialize time parameters
    def initialize_time(self):
        self.initial_time = pd.Timestamp.now()
        try:
            run_time_delta = pd.Timedelta(self.run_time, self.run_time_unit)
        except ValueError:
            run_time_delta = pd.Timedelta(self.run_time, 'h')
            raise ValueError(f"Invalid time unit: {self.run_time_unit}. Expected 'min', 'h', or 'd'. Using 'h'.")
        
        self.end_time = self.initial_time + run_time_delta

    # Define and initialize the model, optimizer, and loss function
    def initialize_model(self):
        input_size = len(self.price_data) - 1   # Number of input columns

        self.model = CryptoModel(model_class=self.model_class, input_size=input_size, hidden_size=128, output_size=1, verbose=self.verbose) # Model instance

    # Get live data
    def get_live_data(self):
        live_data = self.coinbase_api.get_live_data()
        return live_data
    
    def get_historical_data(self, train_period, train_period_unit):
        try:
            train_period_delta = pd.Timedelta(train_period, train_period_unit)
        except ValueError:
            train_period_delta = pd.Timedelta(train_period, 'h')
            raise ValueError(f"Invalid time unit: {train_period_unit}. Expected 'h', 'd', 'w', or 'm'. Using 'm'.")
        
        historical_data = self.coinbase_api.get_historical_data(granularity=self.trade_interval, train_period=train_period_delta)

        return historical_data
    # Add indicators to the data
    def concat_indicators(self, data):
        if not self.indicators: pass    # Pass if indicators toggle is false

        # Compute the indicators
        data['sma'] = ta.sma(data['close'], length=20)                            # Simple Moving Average
        data['rsi'] = ta.rsi(data['close'], length=14)                            # Relative Strength Index
        data['macd'], _, _ = ta.macd(data['close'], fast=12, slow=26, signal=9)   # Moving Average  Convergence Divergence

        data.dropna(inplace=True)  # Drop rows with NaN values
        data.reset_index(drop=True, inplace=True)  # Reset the index

    # Update the model using the latest price data
    def update_model(self, data):
        self.model.update_model(data)    # Call update method in CryptoLSTM class
        
    # Use the model to make a price prediction
    def predict(self):
        predicted_price, confidence = self.model.predict(data=self.price_data, hidden=self.hidden)
        return predicted_price, confidence
    
    # Get order amount
    def get_order_amount(self):
        balance = self.coinbase_api.get_wallet_balance()    # Wallet balance
        order = balance * self.order_p   # Amount to buy/sell
        return order, balance
        
    # Use the model to make a trade decision
    #TODO: if balance == 0, hold
    def make_trade_decision(self, predicted_price, current_price, confidence, order):

        slippage = order * predicted_price * self.slippage_p    # Slippage ($)
        fees = order * predicted_price * self.fees_p            # Fees ($)

        buy_threshold = self.buy_threshold + slippage + fees
        sell_threshold = self.sell_threshold + slippage + fees

        percentage_diff = (predicted_price - current_price) / current_price
        
        if percentage_diff > buy_threshold and confidence > self.confidence_threshold:
            return "BUY"
        elif percentage_diff < -sell_threshold and confidence > self.confidence_threshold:
            return "SELL"
        else:
            return "HOLD"
    
    # Execute a buy order using Coinbase API
    #TODO: Move contents to CoinbaseAPI class, just call the method here
    def buy(self, order):
        auth_client = cbpro.AuthenticatedClient(*self.coinbase_api.get_cb_credentials())
        order = auth_client.place_market_order(product_id=self.product_id, side='buy', funds=order)

    # Execute a sell order using Coinbase API
    def sell(self, order):
        auth_client = cbpro.AuthenticatedClient(*self.coinbase_api.get_cb_credentials())
        order = auth_client.place_market_order(product_id=self.product_id, side='sell', funds=order)

    # Execute a trade based on the current trade decision
    def execute_trade(self, trade_decision, order):
        if trade_decision == 'BUY':
            self.buy(order)
        elif trade_decision == 'SELL':
            self.sell(order)

    # Log order
    def log_order(self, trade_decision, order, predicted_price, confidence, current_price, balance):
        # Create a dictionary of the order details
        order = {
            'trade_decision': trade_decision,
            'order': order,
            'predicted_price': predicted_price,
            'confidence': confidence,
            'current_price': current_price,
            'current_balance': balance,
            'timestamp': pd.Timestamp.now()
        }

        # Append the order details to the order log
        self.order_log = self.order_log.append(order, ignore_index=True)

    # Print current status of the algorithm
    def print_status(self):
        if self.verbose: print(self.order_log.iloc[-1])

    # Train and run
    def train_run(self, historical_period=1, historical_period_unit='m', batch_size=32, epochs=10, seq_length=10):
        self.train(historical_period=historical_period, 
                   historical_period_unit=historical_period_unit, 
                   batch_size=batch_size, 
                   epochs=epochs, 
                   seq_length=seq_length)
        self.run()

    # Train the model on historical data
    #TODO: Change all time periods to dates instead of "1 months"
    def train(self, historical_period=1, historical_period_unit='m', batch_size=32, epochs=10, seq_length=10):
        # Get historical data on training period
        self.train_data = self.get_historical_data(granularity=self.trade_interval,
                                                   historical_period=historical_period, 
                                                   historical_period_unit=historical_period_unit)
        self.concat_indicators(self.train_data) # Concat_indicators

        self.model.train(data=self.train_data, batch_size=batch_size,epochs=epochs, seq_length=seq_length)

    # Start the live trading loop
    #TODO: Change all time periods to dates instead of "1 months"
    def run(self):

        self.initialize_time()          # Initialize time parameters

        # Initialize the current time and end time
        current_time = self.initial_time
        end_time = self.end_time

        # Loop until end time is reached
        while current_time < end_time:
            live_data = self.get_live_data()               # Get the live data
            self.concat_indicators(live_data)              # Add indicators to the data
            self.update_model(self.price_data)                          # Update the model
            predicted_price, confidence = self.predict(self.price_data) # Predicted price
            current_price = live_data['close'][0]                       # Current price
            order, balance = self.get_order_amount()                    # Order amount
            trade_decision = self.make_trade_decision(predicted_price=predicted_price,  # Make a trade decision
                                                      current_price=current_price, 
                                                      confidence=confidence, 
                                                      order=order)
            self.execute_trade(trade_decision, order)                # Execute trade

            self.log_order(trade_decision, order, predicted_price, confidence, current_price, balance)   # Log order
            self.print_status() # Print the current status

            time.sleep(self.trade_interval) # Wait for the trade interval before repeating the loop
            current_time = pd.Timestamp.now()   # Update the current time
            #TODO: Account for time it takes to trade?
    

