"""
LSTM class
"""

# Import Libraries
import torch
import torch.nn as nn


# Import Libraries
import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, fc_hidden_dim, use_hn, dropout, maintain_state=False):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc_hidden_dim = fc_hidden_dim
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.use_hn = use_hn
        self.dropout = dropout
        self.maintain_state = maintain_state  # New attribute to toggle state maintenance
        
        # Initialize hidden states if maintain_state is True
        if self.maintain_state:
            self.h0 = None
            self.c0 = None

        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True, dropout=self.dropout)
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=self.fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.fc_hidden_dim, out_features=output_dim)
        )

    def forward(self, x):
        if not self.maintain_state:
            # Initialize hidden states at each forward pass
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        else:
            # Use stored states if they exist, else initialize
            if self.h0 is None or self.c0 is None:
                self.h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
                self.c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            h0, c0 = self.h0, self.c0

        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Update stored states if maintain_state is True
        if self.maintain_state:
            self.h0, self.c0 = hn.detach(), cn.detach()

        if self.use_hn:
            hn = hn.view(-1, self.hidden_dim)
            out = self.fc(hn)
        else:
            out = out[:, -1, :]
            out = self.fc(out)

        return out

    