## Test Cases for CryptoTrader
# Harrison Floam, 23 April 2023

# Import Libaries and Modules
import unittest

from testcryptotrader import TestCryptoTrader

# Testing data
filepath = '/data/btc_daily.csv'


def test_case_1():
    # Define CryptoTrader parameters
    params = {
        'initial_balance': 100,
        'trade_interval': 60,
        'run_time': 60,
        'run_time_unit': 'min',
        'model_class': 'CryptoLSTM',
        'product_id': 'BTC',
        'buy_threshold': 0.02,
        'sell_threshold': 0.02,
        'order_p': 0.1,
        'confidence_threshold': 0.80,
        'slippage_p': 0.005,
        'fees_p': 0.005,
        'indicators': True,
        'coinbase_api': None,
        'verbose': True,
        'filepath': filepath
    }
    
    trader = TestCryptoTrader(**params)

if __name__ == '__main__':
    test_case_1()