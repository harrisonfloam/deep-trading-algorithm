## Crypto Trading Algorithm - Test File
# Harrison Floam, 10 April 2023

# Import
import unittest
from unittest.mock import Mock

from cryptotrader import CryptoLSTM, CoinbaseAPI, CryptoTrader, initialize_params


class TestCryptoTrader(unittest.TestCase):
    def setUp(self):
        # Create a mock for CoinbaseAPI
        self.mock_api = Mock()
        self.mock_api.get_wallet_balance.return_value = 1000
        self.mock_api.get_live_data.return_value = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', end='2023-02-01', freq='D'),
            'price': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        })

        # Create an instance of the CryptoTrader class
        self.trader = CryptoTrader(
            initial_balance=1000, 
            trade_interval=1,
            product_id='BTC',
            buy_threshold=0.02,
            sell_threshold=0.02,
            order_p=0.1,
            confidence_threshold=0.80,
            slippage_p=0.005,
            fees_p=0.005,
            verbose=False
        )

        # Set the CoinbaseAPI object in the CryptoTrader instance to the mock
        self.trader.coinbase_api = self.mock_api

    def test_run(self):
        # Set the initial and end times
        self.trader.initial_time = pd.Timestamp('2023-01-01')
        self.trader.end_time = pd.Timestamp('2023-01-31')

        # Run the CryptoTrader instance
        self.trader.run()

        # Verify that the wallet balance is greater than or equal to zero
        self.assertGreaterEqual(self.trader.initial_balance, 0)

