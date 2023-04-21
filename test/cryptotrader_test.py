## Crypto Trading Algorithm - Test File
# Harrison Floam, 10 April 2023


# Import Libraries
import pandas as pd
import unittest
from unittest.mock import Mock

# Import Modules and Packages
from src.cryptotrader import CryptoTrader


class TestCryptoTrader(CryptoTrader):
    """
    A wrapper class for CryptoTrader that allows testing and creates additional testing methods.
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
    def __init__(self):
        super().__init__()      # Inherit from parent class
        self.test = True        # Set test flag to true

    # Train and run
    def test_train_run(self, test_train_data, batch_size=32, epochs=10, seq_length=10):
        self.test_train(test_train_data=test_train_data, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        seq_length=seq_length)
        self.test_run()

    # Train model with test data
    def test_train(self, test_train_data, batch_size=32, epochs=10, seq_length=10):
        if not self.test: pass  # Pass if not in test mode

        self.concat_indicators(test_train_data) # Concat_indicators

        self.model.train(data=test_train_data, batch_size=batch_size,epochs=epochs, seq_length=seq_length)

    # Run live trading loop with test data
    #TODO: Figure this out
    def test_run(self, test_data):
        if not self.test: pass  # Pass if not in test mode
        
        self.initialize_time()  # Initialize time parameters

        # Initialize the current time and end time
        current_time = self.initial_time
        end_time = self.end_time

        for i, row in test_data.iterrows():
            # Format data as a dataframe
            data = pd.DataFrame([row], columns=test_data.columns)

            self.concat_indicators(data)  # Add indicators to the data
            self.update_model(data)  # Update the model
            predicted_price, confidence = self.predict()  # Predicted price
            current_price = data['close'][0]  # Current price
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
