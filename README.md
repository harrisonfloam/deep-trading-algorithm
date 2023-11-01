# deep-trading-algorithm

Work in progress.

A Python project that explores deep learning methods for financial forecasting and trading.

## Outline

- - [`archive`](https://github.com/harrisonfloam/deep-trading-algorithm/tree/main/archive): archived files from previous iterations
- [`checkpoints`](https://github.com/harrisonfloam/deep-trading-algorithm/tree/main/checkpoints): model weights and raytune checkpoints
- [`config`](https://github.com/harrisonfloam/deep-trading-algorithm/tree/main/config): environment and other configs
- [`data`](https://github.com/harrisonfloam/deep-trading-algorithm/tree/main/data): source data .csv's
- [`logs`](https://github.com/harrisonfloam/deep-trading-algorithm/tree/main/logs): tensorboard training logs
- [`notebooks`](https://github.com/harrisonfloam/deep-trading-algorithm/tree/main/notebooks): jupyter notebooks for exploration and analysis
- [`src/`](https://github.com/harrisonfloam/deep-trading-algorithm/tree/main/src): project source
  - [`data`](https://github.com/harrisonfloam/deep-trading-algorithm/tree/main/src/data): classes and utilities for data preprocessing and loading
  - [`model`](https://github.com/harrisonfloam/deep-trading-algorithm/tree/main/src/model): model class definitions
  - [`predict`](https://github.com/harrisonfloam/deep-trading-algorithm/tree/main/src/predict): classes and utilities for inference with trained models
  - [`train`](https://github.com/harrisonfloam/deep-trading-algorithm/tree/main/src/train): classes and utilities for training models
  - [`tune`](https://github.com/harrisonfloam/deep-trading-algorithm/tree/main/src/tune): classes and utilities for hyperparameter tuning and optimization


## Planned Features

- Classification, regression, and ensemble models
- Training with Tensorboard
- Tuning with RayTune
- Attribution visualization
- Trading strategy with supervised learning
- Trading strategy with reinforcement learning
- Containerization and deployment
- Command line execution

## Usage

## Notes

## Environment

Create a conda environment and activate it.
```
conda env create -f config/environment.yml
conda activate time-series
```
