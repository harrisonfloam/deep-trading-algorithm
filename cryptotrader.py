## Crypto Trading Algorithm
# Harrison Floam, 10 April 2023
# crypto-trading-algorithm

# Import Libraries
import requests
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pandas_ta as ta
import configparser
import time
import cbpro

# Define the Model Class
class CryptoLSTM(nn.Module):
    def __init__(self, input_size, indicator_size, hidden_size, output_size):
        super(CryptoLSTM, self).__init__()
        # Define the layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)      # LSTM layer
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)                   # Fully-connected layer 1
        self.fc2 = nn.Linear(hidden_size//2 + indicator_size, output_size)  # Fully-connected layer 2
        self.sigmoid = nn.Sigmoid()                                         # Sigmoid activation function
    # Define the forward function
    def forward(self, x, hidden, indicator):
        out, hidden = self.lstm(x, hidden)                  # Pass input and previous hidden state through LSTM layer
        out = self.fc1(out[:, -1, :])                       # Pass output of LSTM layer through first fully connected layer
        out = self.sigmoid(out)                             # Apply sigmoid activation function
        out = self.fc2(torch.cat((out, indicator), dim=1))  # Concatenate output of first fully connected layer with indicator data and pass through second fully connected layer
        out = self.sigmoid(out)                             # Apply sigmoid activation function
        return out, hidden

class CoinbaseAPI():
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
    
    def get_historical_data(self):
        pass

class CryptoTrader:
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
    def initialize_model(self):
        indicator_size = len(self.price_data.columns) - 2 # Number of indicator columns

        self.model = CryptoLSTM(input_size=1, indicator_size=indicator_size, hidden_size=128, output_size=1) # Model instance
        self.criterion = nn.MSELoss()                                # MSE loss function
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)    # Adam optimizer

    
    # Add indicators to the data
    def concat_indicators(self):
        # Compute the indicators
        self.price_data['sma'] = ta.sma(self.price_data['price'], length=20)                            # Simple Moving Average
        self.price_data['rsi'] = ta.rsi(self.price_data['price'], length=14)                            # Relative Strength Index
        self.price_data['macd'], _, _ = ta.macd(self.price_data['price'], fast=12, slow=26, signal=9)   # Moving Average  Convergence Divergence

        self.price_data.dropna(inplace=True)  # Drop rows with NaN values
        self.price_data.reset_index(drop=True, inplace=True)  # Reset the index

    # Update the model using the latest price data
    def update_model(self):
        input_seq = self.price_data.iloc[-2:-1, 1:].values       # Double check
        target_seq = self.price_data.iloc[-1:, 1].values.reshape(-1, 1)

        input_seq = torch.tensor(input_seq).unsqueeze(1).float()    # Convert input to tensor
        target_seq = torch.tensor(target_seq).float()               # Convert target to tensor

        self.optimizer.zero_grad()   # Clear the gradients from the optimizer

        output, self.hidden = self.model(input_seq, self.hidden)    # Pass the input sequence and previous hidden state through the model
        loss = self.criterion(target_seq, output)                   # Compute the loss between the predicted and actual values

        loss.backward()         # Backpropagate loss
        self.optimizer.step()   # Update model parameters
        
    # Use the model to make a price prediction
    def predict(self):
        self.model.eval()
        with torch.no_grad():
            input_seq = self.price_data.iloc[-1:, 1:].values
            input_seq = torch.tensor(input_seq).unsqueeze(1).float()
            output, _ = self.model(input_seq, self.hidden)
            predicted_price = output.item()
            confidence = 1.0 - self.criterion(output, input_seq[:, -1:, :]).item()
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

    # Train the model on historical data
    def train(self, historical_period=1, historical_period_unit='m', batch_size=32, epochs=10):
        # Initialize model
        # Get historical data
        # Concat_indicators
        # Train
        pass
    # Start the live trading loop
    def run(self):

        self.initialize_time()          # Initialize time parameters
        self.initialize_model()         # Initialize the model
        balance = self.initial_balance  # Initialize the wallet balance

        # Initialize the current time and end time
        current_time = self.initial_time
        end_time = self.end_time

        # Get historical data

        # Loop until end time is reached
        while current_time < end_time:
            live_data = self.coinbase_api.get_live_data()               # Get the live data
            self.price_data = self.concat_indicators(live_data)         # Add indicators to the data
            self.update_model(self.price_data)                          # Update the model
            predicted_price, confidence = self.predict(self.price_data) # Predicted price
            current_price = live_data['price'][0]                       # Current price
            order, balance = self.get_order_amount()                    # Order amount
            trade_decision = self.make_trade_decision(predicted_price=predicted_price, 
                                                      current_price=current_price, 
                                                      confidence=confidence, 
                                                      order=order)   # Make a trade decision
            self.execute_trade(trade_decision, order)  # Execute trade

            self.log_order(trade_decision, order, predicted_price, confidence, current_price, balance)   # Log order
            self.print_status() # Print the current status

            time.sleep(self.trade_interval) # Wait for the trade interval before repeating the loop
            current_time = pd.Timestamp.now()   # Update the current time


