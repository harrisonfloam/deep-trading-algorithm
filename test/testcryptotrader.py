## Crypto Trading Algorithm - Test File
# Harrison Floam, 10 April 2023


# Import Libraries
import pandas as pd
import numpy as np
import unittest
from unittest.mock import Mock
from sklearn.model_selection import train_test_split

# Import Modules and Packages
from src.cryptotrader import CryptoTrader
from src.cryptomodel import CryptoModel
from src.coinbase import CoinbaseAPI


class TestCryptoTrader(CryptoTrader):
    """
    A wrapper class for CryptoTrader that allows use of local data and creates additional testing methods.
    To test, call test_train(), then test_run(), or test_train_run().

    ### Methods:
    -----------
    - test_train()
        Train the model with test data
    - test_run()
        Run the live trading loop with test data
    - test_train_run()
        Train and run with test data
    """
    def __init__(self, filepath, **kwargs):
        super().__init__(**kwargs)      # Inherit from parent class
        self.test = True        # Set test flag to true
        self.verbose = True     # Set verbose flag to true
        
        self.filepath = filepath                # Filepath to testing data
        self.test_train_data = pd.DataFrame()   # Train set
        self.test_data = pd.DataFrame()         # Test set

    # Read testing dataset
    def get_test_data(self):
        df = pd.read_csv(self.filepath)  # Read file to dataframe

        self.test_train_data, self.test_data = train_test_split(df, test_size=0.2, shuffle=False)

    # Train and run
    def test_train_run(self, batch_size=32, epochs=10, seq_length=10):
        self.test_train(filepath=self.filepath, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        seq_length=seq_length)
        self.test_run()

    # Train model with test data
    def test_train(self, batch_size=32, epochs=10, seq_length=10):
        if not self.test: pass  # Pass if not in test mode

        self.get_test_data()
        self.test_train_data = self.concat_indicators(self.test_train_data)

        self.initialize_model(self.test_train_data)     # Initialize model

        self.model.train(data=self.test_train_data, batch_size=batch_size,epochs=epochs, seq_length=seq_length)

    # Run live trading loop with test data
    #TODO: Figure this out
    def test_run(self):
        if not self.test: pass  # Pass if not in test mode
        
        self.initialize_time()  # Initialize time parameters

        # Initialize the current time and end time
        current_time = self.initial_time
        end_time = self.end_time

        for i, row in self.test_data.iterrows():
            # Format data as a dataframe
            data = pd.DataFrame([row], columns=self.test_data.columns)  #TODO: what is this doing?

            self.concat_indicators(data)  # Add indicators to the data  #TODO: concat indicators needs to change
            self.update_model(data)  # Update the model #TODO: Check on this method
            predicted_price, confidence = self.predict()  # Predicted price #TODO: Check on this method
            current_price = data['close'][0]  # Current price   #TODO: Check on this method
            order, balance = self.get_order_amount()  # Order amount
            trade_decision = self.make_trade_decision(predicted_price=predicted_price,  # Make a trade decision
                                                       current_price=current_price,
                                                       confidence=confidence,
                                                       order=order)
            self.execute_trade(trade_decision, order)  # Execute trade
            self.log_order(trade_decision, order, predicted_price, confidence, current_price, balance)  # Log order
            self.print_status()  # Print the current status

            current_time += pd.Timedelta(seconds=self.trade_interval)
            if current_time >= end_time:
                break
