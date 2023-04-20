## Coinbase API Interaction
# Harrison Floam, 18 April 2023

# Import
import cbpro
import configparser
import requests
import datetime
import pandas as pd

class CoinbaseAPI():
    """
    A class for interacting with the Coinbase Pro API to retrieve account information and real-time cryptocurrency prices.

    ### Methods:
    -----------
    - get_credentials(self)
        Retrieves the API key, secret key, and passphrase from the config.ini file.
    - get_wallet_balance(self)
        Retrieves the account balance of the user's wallet on Coinbase.
    - get_live_data(self)
        Retrieves the real-time price data of a specified cryptocurrency on Coinbase.
    - get_historical_data(self)
        Retrieves the historical price data of a specified cryptocurrency on Coinbase.
    """

    #TODO: Confirm actual CB API
    #TODO: Figure out how to store secret key

    def __init__(self, product_id):
        self.base_url = 'https://api.coinbase.com/v2'
        self.product_id = product_id
        self.key, self.secret, self.passphrase = self.get_credentials()

    # Get Coinbase credentials from .ini file
    def get_credentials(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        key = config.get('coinbase', 'key')
        secret = config.get('coinbase', 'secret')
        passphrase = config.get('coinbase', 'passphrase')
        return key, secret, passphrase

    # Get current Coinbase wallet balance
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

    # Get live price data
    def get_live_data(self):
        endpoint = f'{self.base_url}/prices/{self.product_id}-USD/spot'

        response = requests.get(endpoint)
        data = response.json()['data']

        timestamp = pd.to_datetime(data['timestamp']).tz_localize(None)

        price_data = pd.DataFrame({
            'timestamp': [timestamp],
            'price': [float(data['price'])],
            'volume': [float(data['volume'])],
            'low': [float(data['low'])],
            'high': [float(data['high'])],
            'open': [float(data['open'])],
            'close': [float(data['last_trade_price'])]
        })

        return price_data
    
    # Get historical price data
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
        
        return df
