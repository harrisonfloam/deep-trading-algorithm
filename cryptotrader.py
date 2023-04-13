## Crypto Trading Algorithm
# Harrison Floam, 10 April 2023

# Import Libraries
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas_ta as ta
import configparser
import time
import cbpro
import datetime
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CryptoTrader:
    """
    A class for creating and training a LSTM-based cryptocurrency trading algorithm. 
    Initialize, then call train(), then run(), or train_run().

    Methods:
        train(self, data, batch_size=32, epochs=10)
            Trains the LSTM model on the given historical price data.
        run()
            Starts the live trading loop.
        train_run()
            Train the model and start the live trading loop in one call.
    """
    def __init__(self, initial_balance, trade_interval, run_time, 
                 run_time_unit='h', product_id='BTC', buy_threshold=0.02, sell_threshold=0.02,
                 order_p=0.1, confidence_threshold=0.80, slippage_p=0.005, fees_p=0.005, verbose=False):
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
        self.price_data = pd.DataFrame()    # Price dataframe
        self.model = None                   # LSTM model
        self.criterion = None               # Model criterion
        self.optimizer = None               # Model optimizer
        self.hidden = None                  # Hidden layers

        # Initialize Coinbase API parameters
        self.coinbase_api = CoinbaseAPI(product_id=product_id)

        # Initialize time parameters
        self.run_time = run_time
        self.run_time_unit = run_time_unit

        # Initialize tracking parameters
        self.order_log = pd.DataFrame()
        self.verbose = verbose

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
    #TODO: Move criterion and optimizer to CryptoLSTM
    def initialize_model(self):
        indicator_size = len(self.price_data.columns) - 2 # Number of indicator columns

        self.model = CryptoLSTM(input_size=1, indicator_size=indicator_size, hidden_size=128, output_size=1, verbose=self.verbose) # Model instance

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
        # Compute the indicators
        data['sma'] = ta.sma(data['price'], length=20)                            # Simple Moving Average
        data['rsi'] = ta.rsi(data['price'], length=14)                            # Relative Strength Index
        data['macd'], _, _ = ta.macd(data['price'], fast=12, slow=26, signal=9)   # Moving Average  Convergence Divergence

        data.dropna(inplace=True)  # Drop rows with NaN values
        data.reset_index(drop=True, inplace=True)  # Reset the index

    # Update the model using the latest price data
    # TODO: Use create_sequences() here
    def update_model(self):
        input_seq = self.price_data.iloc[-2:-1, 1:].values       # Double check
        target_seq = self.price_data.iloc[-1:, 1].values.reshape(-1, 1)

        input_seq = torch.tensor(input_seq).unsqueeze(1).float()    # Convert input to tensor
        target_seq = torch.tensor(target_seq).float()               # Convert target to tensor

        self.optimizer.zero_grad()   # Clear the gradients from the optimizer

        output, self.hidden = self.model(input_seq, self.hidden)    # Pass the input sequence and previous hidden state through the model
        loss = self.model.criterion(target_seq, output)                   # Compute the loss between the predicted and actual values

        loss.backward()         # Backpropagate loss
        self.model.optimizer.step()   # Update model parameters
        
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
        print(self.order_log.iloc[-1])

    # Train and run
    def train_run(self, historical_period=1, historical_period_unit='m', batch_size=32, epochs=10, seq_length=10):
        self.train(historical_period=historical_period, 
                   historical_period_unit=historical_period_unit, 
                   batch_size=batch_size, 
                   epochs=epochs, 
                   seq_length=seq_length)
        self.run()

    # Train the model on historical data
    def train(self, historical_period=1, historical_period_unit='m', batch_size=32, epochs=10, seq_length=10):
        self.train_data = self.get_historical_data(granularity=self.trade_interval, # Get historical data
                                                   historical_period=historical_period, 
                                                   historical_period_unit=historical_period_unit)
        self.concat_indicators(self.train_data) # Concat_indicators

        self.model.train(data=self.train_data, batch_size=batch_size,epochs=epochs, seq_length=seq_length)

    # Start the live trading loop
    #TODO: Change all time periods to dates instead of "1 months"
    def run(self):

        self.initialize_time()          # Initialize time parameters
        balance = self.initial_balance  # Initialize the wallet balance

        # Initialize the current time and end time
        current_time = self.initial_time
        end_time = self.end_time

        # Loop until end time is reached
        while current_time < end_time:
            live_data = self.get_live_data()               # Get the live data
            self.concat_indicators(live_data)              # Add indicators to the data
            self.update_model(self.price_data)                          # Update the model
            predicted_price, confidence = self.predict(self.price_data) # Predicted price
            current_price = live_data['price'][0]                       # Current price
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


class CryptoLSTM(nn.Module):
    """
    A class for creating an LSTM-based cryptocurrency trading algorithm.

    Methods:
        __init__(self, input_size, indicator_size, hidden_size, output_size)
            Initializes the LSTM model with the specified input, indicator, and output sizes.
        create_sequences(self, data, seq_length=10)
            Converts the historical price data to sequences and labels for training.
        train(self, data, batch_size=32, epochs=10)
            Trains the LSTM model on the given historical price data.
        predict(self, sequence)
            Predicts the next price in a given sequence using the trained LSTM model.

    """
    def __init__(self, input_size, indicator_size, hidden_size, output_size, verbose=False):
        super(CryptoLSTM, self).__init__()
        # Define the layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)      # LSTM layer
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)                   # Fully-connected layer 1
        self.fc2 = nn.Linear(hidden_size//2 + indicator_size, output_size)  # Fully-connected layer 2
        self.sigmoid = nn.Sigmoid()                                         # Sigmoid activation function

        # Define model parameters
        self.criterion = nn.MSELoss()                                # MSE loss function
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)     # Adam optimizer

        # Define other parameters
        self.verbose = verbose  # Verbose debug flag

    # Define the forward function
    def forward(self, x, hidden, indicator):
        out, hidden = self.lstm(x, hidden)                  # Pass input and previous hidden state through LSTM layer
        out = self.fc1(out[:, -1, :])                       # Pass output of LSTM layer through first fully connected layer
        out = self.sigmoid(out)                             # Apply sigmoid activation function
        out = self.fc2(torch.cat((out, indicator), dim=1))  # Concatenate output of first fully connected layer with indicator data and pass through second fully connected layer
        out = self.sigmoid(out)                             # Apply sigmoid activation function
        return out, hidden
    
    def create_sequences(self, data, seq_length):
        """
        Create sequences for training/evaluation
        """
        sequences = []
        targets = []

        # Extract price and indicator data
        price_data = data[['price']]
        indicator_data = data.drop(columns=['price'])

        # Iterate over the data to create sequences
        for i in range(seq_length, len(data)):
            sequence = price_data[i - seq_length:i]
            target = data[i:i+1]['price'].values[0]

            # Add indicator values to the sequence
            indicators = indicator_data[i - seq_length:i].values
            sequence = np.hstack([sequence.values, indicators])

            sequences.append(sequence)
            targets.append(target)

        # Convert lists to tensors
        sequences = torch.tensor(sequences, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)

        return sequences, targets
    
    def train(self, data, batch_size=32, epochs=10, seq_length=10):
        sequences, labels = self.create_sequences(data=data, seq_length=seq_length)  # Convert training data to sequences and labels

        # Create DataLoader
        dataset = CryptoDataset(sequences, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Train the model
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(dataloader):
                inputs, labels = data
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            if self.verbose: print(f'Epoch {epoch+1} loss: {running_loss/len(dataloader):.6f}') # Print if verbose

    def predict(self, data, hidden):
        self.eval()
        with torch.no_grad():
            input_seq = data.iloc[-1:, 1:].values
            input_seq = torch.tensor(input_seq).unsqueeze(1).float()
            output, _ = self(input_seq, hidden)
            predicted_price = output.item()
            confidence = 1.0 - self.criterion(output, input_seq[:, -1:, :]).item()
        return predicted_price, confidence


class CryptoDataset(Dataset):
    """
    A class for creating a PyTorch dataset from sequences and labels.
    """
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sequences[index], self.labels[index]


class CoinbaseAPI():
    """
    A class for interacting with the Coinbase Pro API to retrieve account information and real-time cryptocurrency prices.

    Methods:
        get_credentials(self)
            Retrieves the API key, secret key, and passphrase from the config.ini file.
        get_wallet_balance(self)
            Retrieves the account balance of the user's wallet on Coinbase.
        get_live_data(self)
            Retrieves the real-time price of a specified cryptocurrency on Coinbase.
        get_historical_data(self)
            Retrieves the historical price data of a specified cryptocurrency on Coinbase.
    """
    def __init__(self, product_id):
        self.base_url = 'https://api.coinbase.com/v2'
        self.product_id = product_id
        self.key, self.secret, self.passphrase = self.get_credentials()

    def get_credentials(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        key = config.get('coinbase', 'key')
        secret = config.get('coinbase', 'secret')
        passphrase = config.get('coinbase', 'passphrase')
        return key, secret, passphrase

    def get_wallet_balance(self):
        endpoint = f'{self.base_url}/accounts'
        headers = {
            'CB-ACCESS-KEY': self.key,
            'CB-ACCESS-SIGN': self.secret,
            'CB-ACCESS-PASSPHRASE': self.passphrase,
            'CB-VERSION': '2023-04-10'
        }
        response = requests.get(endpoint, headers=headers)
        data = response.json()
        balance = float(data['data'][0]['balance']['amount'])
        return balance

    def get_live_data(self):
        endpoint = f'{self.base_url}/prices/{self.product_id}-USD/spot'
        params = {'currency': {self.product_id}}
        response = requests.get(endpoint, params=params)
        data = response.json()['data']
        timestamp = pd.to_datetime(data['timestamp']).tz_localize(None)
        price = float(data['amount'])
        price_data = pd.DataFrame({'timestamp': [timestamp], 'price': [price]})
        return price_data
    
    def get_historical_data(self, granularity, train_period):
        end_time = datetime.now()
        start_time = end_time - train_period
        
        endpoint = f"{self.base_url}/products/{self.product_id}/candles"
        params = {
            'start': start_time.isoformat(),
            'end': end_time.isoformat(),
            'granularity': granularity
        }
        response = requests.get(endpoint, params=params)
        data = response.json()
        df = pd.DataFrame(data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index('time')
        df = df[['open']]
        
        return df
