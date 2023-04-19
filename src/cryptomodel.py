## Crypto Model Class
# Harrison Floam, 18 April 2023

from models.crypto_lstm import CryptoLSTM

class CryptoModel:
    """A wrapper class for different cryptocurrency trading models.
    
    ### Parameters:
    -----------
    - model_class: class
        The trading model class to be used (e.g. CryptoLSTM).
    - **kwargs:
        Additional arguments to be passed to the model class constructor.
    """
    def __init__(self, model_class, **kwargs):
        super().__init__()
        self.model = model_class(**kwargs)