## Crypto Model Class
# Harrison Floam, 18 April 2023


class CryptoModel:
    """A wrapper class for different cryptocurrency trading models.
    
    ### Parameters:
    -----------
    - model_class_name: string
        The trading model class to be used (e.g. CryptoLSTM).
    - **kwargs:
        Additional arguments to be passed to the model class constructor.
    """
    def __init__(self, model_class_name, **kwargs):
        model_module = __import__('models.' + model_class_name.lower(), fromlist=[model_class_name])
        model_class = getattr(model_module, model_class_name)
        self.model = model_class(**kwargs)

    def train(self, *args, **kwargs):
        self.model.train(*args, **kwargs)

    def predict(self, *args, **kwargs):
        self.model.predict(*args, **kwargs)

    def update_model(self, *args, **kwargs):
        self.model.predict(*args, **kwargs)
        