## Crypto LSTM Model Class
# Harrison Floam, 18 April 2023

# Import
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CryptoLSTM(nn.Module):
    """
    A class for creating an LSTM-based cryptocurrency trading algorithm.

    ### Methods:
    -----------
    - __init__(self, input_size, hidden_size, output_size)
        Initializes the LSTM model with the specified input, and output sizes.
    - create_sequences(self, data, seq_length=10)
        Converts the historical price data to sequences and labels for training.
    - train(self, data, batch_size=32, epochs=10)
        Trains the LSTM model on the given historical price data.
    - predict(self, sequence)
        Predicts the next price in a given sequence using the trained LSTM model.

    """
    #TODO: Is this model set up correctly?
    def __init__(self, input_size, hidden_size, output_size=1, verbose=False):
        super(CryptoLSTM, self).__init__()
        # Define the layers
        self.hidden_size = hidden_size
        self.hidden = (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))  # Hidden layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)      # LSTM layer
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)                   # Fully-connected layer 1
        self.fc2 = nn.Linear(hidden_size//2 + input_size, output_size)      # Fully-connected layer 2
        self.sigmoid = nn.Sigmoid()                                         # Sigmoid activation function

        # Define model parameters
        self.criterion = nn.MSELoss()                                # MSE loss function
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)     # Adam optimizer

        # Define other parameters
        self.verbose = verbose  # Verbose debug flag

    # Define the forward function
    # FIXME: Hidden layer is the wrong size? Does training/predicting need different hidden sizes?
    def forward(self, x, hidden):
        out, self.hidden = self.lstm(x, hidden)             # Pass input and previous hidden state through LSTM layer
        out = self.fc1(out[:, -1, :])                       # Pass output of LSTM layer through first fully connected layer
        out = self.sigmoid(out)                             # Apply sigmoid activation function
        out = self.fc2(out)                                 # Pass output of first fully connected layer through second fully connected layer
        out = self.sigmoid(out)                             # Apply sigmoid activation function
        return out, hidden
    
    #TODO: Do I need a backward() or is it part of the model

    # Create tensor sequences for model input
    def create_sequences(self, data, seq_length):
        """
        Create sequences for training/evaluation
        """
        #TODO: What even is seq_length?
        sequences = []
        targets = []

        # Iterate over the data to create sequences
        for i in range(seq_length, len(data)):
            sequence = data.iloc[i - seq_length:i].values
            target = data.iloc[i, 0]

            sequences.append(sequence)
            targets.append(target)

        # Convert lists to numpy arrays
        sequences = np.array(sequences)
        targets = np.array(targets)


        # Convert lists to tensors
        sequences = torch.tensor(sequences, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)

        return sequences, targets
    
    # Train the model
    def train(self, data, seq_length, batch_size=32, epochs=10):
        sequences, labels = self.create_sequences(data=data, seq_length=seq_length)  # Convert training data to sequences and labels

        #TODO: Make sure targets are actually shifting to the next day
        #TODO: Get all variables on the same page - labels, targets, etc.
        # Create DataLoader
        dataset = CryptoDataset(sequences, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Train the model``
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data_load in enumerate(dataloader):
                inputs, labels = data_load
                self.optimizer.zero_grad()
                outputs, _ = self(inputs, self.hidden)
                loss = self.criterion(outputs.squeeze(), labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            if self.verbose: print(f'Epoch {epoch+1} loss: {running_loss/len(dataloader):.6f}') # Print if verbose

    # Query the model
    def predict(self, data, hidden):
        self.eval()     # Toggle evaluation mode
        with torch.no_grad():
            input_seq = data.iloc[-1:, :].values                        # Last row of dataframe (most recent features)
            input_seq = torch.tensor(input_seq).unsqueeze(1).float()    #TODO: do we want to predict with a sequence > 1?
            output, _ = self(input_seq, hidden)
            predicted_price = output.item()
            confidence = 1.0 - self.criterion(output, input_seq[:, -1:, :]).item()
        return predicted_price, confidence
    
    # Update the model with new data
    def update_model(self, data, seq_length=1):
        input_seq, target_seq = self.create_sequences(data=data, seq_length=seq_length)
        self.optimizer.zero_grad()  # Clear the gradients from the optimizer

        output, self.hidden = self(input_seq, self.hidden)  # Pass the input sequence and previous hidden state through the model
        loss = self.criterion(target_seq, output)  # Compute the loss between the predicted and actual values

        loss.backward()  # Backpropagate loss
        self.optimizer.step()  # Update model parameters


class CryptoDataset(Dataset):
    """
    A class for creating a PyTorch dataset from sequences and labels.
    """
    #TODO: Is this needed?
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sequences[index], self.labels[index]