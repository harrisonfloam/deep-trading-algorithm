# crypto-trading-algorithm

Work in progress.

Python project that creates a live trading framework for a chosen cryptocurrency using a PyTorch LSTM neural network model.

## Outline
- src/
    - `__init__.py`
    - cryptotrader.py
    - cryptomodel.py
    - coinbase.py
    - model/
        crypto_lstm.py
- test/
    - `__init__.py`
    - testcryptotrader.py
    - cryptotrader_testcases.py
    - cryptotrader_unitttests.py
    - data/
        - btc_daily.csv
- README.md

## Usage
1. Initialize a CryptoTrader instance with desired parameters. Choose cryptocurrency, initial balance to trade with, total live trading time, trade interval, and trading parameters.
2. Train the model using the train method of the CryptoTrader class. Choose training period and various model parameters.
3. Run the model using the run method. The live trading loop gets live data, updates the model, predicts the future price, makes a trade decision, and loops until the end time is reached.

## Notes
- Training and live data features are current price and indicator values for the previous time step, target is the current price.
- Trade decisions are made by make_trade_decision method in CryptoTrader class. Trades are made when the predicted price, accounting for estimated slippage and fees, is greater or less than the buy or sell threshold and the confidence of the prediction is greater than the confidence threshold.

## Environment
Will be updated in future release.

## VSCode Configuration
settings.json:
<pre>
    {
        "python.analysis.autoImportCompletions": true,
        "python.autoComplete.extraPaths": [
            "./src",
            "./test"
        ],
        "python.analysis.extraPaths": [
            "./src",
            "./test"
        ],
        "terminal.integrated.env.osx": {
            "PYTHONPATH": "${env:PYTHONPATH}:${workspaceFolder}/src",
        },
        "terminal.integrated.env.linux": {
            "PYTHONPATH": "${env:PYTHONPATH}:${workspaceFolder}/src",
        },
        "terminal.integrated.env.windows": {
            "PYTHONPATH": "${env:PYTHONPATH};${workspaceFolder}/src",
        }
    }
</pre>

.env:
<pre>
    PYTHONPATH=${PYTHONPATH}:./src
</pre>
