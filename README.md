# crypto-trading-algorithm

Work in progress.

Python module that creates a live trading framework for a chosen cryptocurrency using a PyTorch LSTM neural network model.

## Usage
1. Initialize a CryptoTrader instance with desired parameters. Choose cryptocurrency, initial balance to trade with, total live trading time, trade interval, and trading parameters.
2. Train the model using the train method of the CryptoTrader class. Choose training period and various model parameters.
3. Run the model using the run method. The live trading loop gets live data, updates the model, predicts the future price, makes a trade decision, and loops until the end time is reached.

Notes:
- Training and live data features are current price and indicator values for the previous time step, target is the current price.
- Trade decisions are made by make_trade_decision method in CryptoTrader class. Trades are made when the predicted price, accounting for estimated slippage and fees, is greater or less than the buy or sell threshold and the confidence of the prediction is greater than the confidence threshold.
