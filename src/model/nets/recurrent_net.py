import torch
import torch.nn as nn

class RNNModel(nn.Module):
    """
    A simple recurrent model (LSTM or GRU) for multistep forecasting.
    Predicts 'horizon' steps ahead from the final hidden state.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 horizon: int, rnn_type: str = 'LSTM', dropout: float = 0.0):
        super(RNNModel, self).__init__()
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon

        # Choose the RNN variant
        if rnn_type.upper() == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout
            )
        elif rnn_type.upper() == 'GRU':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout
            )
        else:
            raise ValueError("rnn_type must be either 'LSTM' or 'GRU'.")

        # Final fully connected layer to map hidden state -> horizon outputs
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          x shape: (batch, seq_len, input_size)
        Returns:
          (batch, horizon)
        """
        out, _ = self.rnn(x)  # out: (batch, seq_len, hidden_size)
        last_out = out[:, -1, :]  # (batch, hidden_size)
        preds = self.fc(last_out)  # (batch, horizon)
        return preds