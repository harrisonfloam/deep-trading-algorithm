## Crypto Model Class
# Harrison Floam, 18 April 2023

from crypto_lstm import CryptoLSTM

class CryptoModel():
    def __init__(self, model_class, **kwargs):
        super().__init__()
        self.model = model_class(**kwargs)