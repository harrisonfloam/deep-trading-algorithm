"""
Tuning script
"""

# Import Libraries
import ray
from ray import tune, train
from ray.tune.search.optuna import OptunaSearch
from functools import partial

# Import Modules
from src.data.data import DataProcessor
from src.model.lstm import LSTM
from src.train.trainer import Trainer
from src.tune.tuner import Tuner
from src.utils import google_colab_handler


def objective(config, data_processor):
    # Handle Google Colab and TPU settings
    running_in_colab, use_TPU, path, device = google_colab_handler(use_TPU=False, verbose=False)
    
    # Create dataloaders
    train_loader, val_loader, _ = data_processor.create_dataloaders(window=config['window'], 
                                                                    batch_size=config['batch_size'],
                                                                    exclude_input_columns=['percent_ret'])

    # Input dimension
    sample_batch = next(iter(train_loader))
    sample_x, _ = sample_batch
    input_dim = sample_x.shape[2] 

    # Initialize model
    model = LSTM(input_dim=input_dim, 
                hidden_dim=config['hidden_dim'], 
                output_dim=1, 
                num_layers=config['num_layers'], 
                fc_hidden_dim=config['fc_hidden_dim'],
                use_hn=False, 
                dropout=config['dropout'],
                maintain_state=True)

    # Initialize trainer
    trainer = Trainer(model=model,
                    device=device,
                    verbose=False,
                    running_in_colab=running_in_colab,
                    use_TPU=use_TPU,
                    use_tensorboard=False)

    # Train model
    trainer.train(train_loader=train_loader,
                val_loader=val_loader,
                epochs=30,
                lr=config['lr'],
                no_change_patience=config['no_change_patience'],
                overfit_patience=100,
                warmup=config['warmup'],
                save_best=False)

    # Report last validation loss to Ray Tune
    last_val_loss = trainer.state.val_losses[-1]
    tune.report({'val_loss': last_val_loss})
    
def prepare_data():
    data_processor = DataProcessor(start='2021-03-30 00:00:00', 
                                end='2021-03-31 00:00:00', 
                                verbose=False)
    data_processor.prepare_data(filename='bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv', 
                                exclude_columns_detrend=['SMA', 'RSI', 'MACD', 'log_ret', 'percent_ret'],
                                exclude_columns_scale=[''])
    return data_processor

search_space = {
    "lr": tune.loguniform(1e-4, 1e-2),
    "batch_size": tune.choice([32, 64, 128]),
    "window": tune.choice([5, 10, 20]),
    "hidden_dim": tune.choice([128, 256, 512]),
    "fc_hidden_dim": tune.choice([128, 256, 512]),
    "dropout": tune.uniform(0.1, 0.5),
    "warmup": tune.choice([0, 50, 100]),
    "no_change_patience": tune.choice([5, 10, 15]),
    "num_layers": tune.choice([1, 2, 3])
}

algo = OptunaSearch()

def tune():
    data_processor = prepare_data()
    objective_func = partial(objective, data_processor)
    tuner = Tuner(objective_func=objective_func,
                  search_space=search_space,
                  algo=algo)
    tuner.tune_model()
    
if __name__ == "__main__":
    tune()