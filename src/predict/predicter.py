"""
Predicter class definition
"""


def predict(self, loader, output_as_df=False):
    self.model.eval()  # Set model to evaluation mode
    start_time = time.time()

    if self.verbose:
        print(f'Predicting --------- Model: {self.model_name}')
    
    t = tqdm(loader, disable=not self.verbose, desc='Predicting')   # Progress bar

    predictions = torch.tensor([], device=self.device)
    timestamps = []
    with torch.no_grad():
        for i, (x, _) in enumerate(t):
            x = x.to(self.device)
            outputs = self.model(x).squeeze()  # Forward pass

            predictions = torch.cat((predictions, outputs), dim=0)
    elapsed_time = time.time() - start_time

    if self.verbose:
        print(f'Time Elapsed: {elapsed_time:.2f}s')
    
    if output_as_df:
        shifted_timestamps = loader.dataset.df.index[loader.dataset.window: loader.dataset.window + len(predictions)]
        return pd.DataFrame({'Timestamp': shifted_timestamps, 'percent_ret': predictions.cpu().numpy()}).set_index('Timestamp')

    return predictions