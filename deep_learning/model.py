"""
PyTorch model blueprint: stacked LSTM + dense head for binary spam/ham (Word2Vec sequence input).
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Defaults aligned with the Word2Vec model (100-dim) and LSTM input padding
DEFAULT_MAX_LEN = 50
DEFAULT_EMBEDDING_DIM = 100


class LSTMSpamModel(nn.Module):
    """
    Stacked LSTM classifier for spam detection.

    Input shape: ``(batch_size, max_len, embedding_dim)`` — one W2V vector per timestep.
    Output shape: ``(batch_size, 1)`` — sigmoid probability of spam.
    """

    def __init__(
        self,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        lstm1_units: int = 64,
        lstm2_units: int = 32,
        dropout_rate: float = 0.3,
        dense_hidden_units: int = 32,
    ):
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm1_units,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(
            input_size=lstm1_units,
            hidden_size=lstm2_units,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(lstm2_units, dense_hidden_units)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_hidden_units, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, embedding_dim)
        out, _ = self.lstm1(x)        # (batch, seq_len, lstm1_units)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)       # (batch, seq_len, lstm2_units)
        out = out[:, -1, :]            # take last timestep: (batch, lstm2_units)
        out = self.dropout2(out)
        out = self.relu(self.fc1(out)) # (batch, dense_hidden_units)
        out = self.sigmoid(self.fc2(out))  # (batch, 1)
        return out


def build_lstm_spam_model(
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    *,
    lstm1_units: int = 64,
    lstm2_units: int = 32,
    dropout_rate: float = 0.3,
    dense_hidden_units: int = 32,
) -> LSTMSpamModel:
    """Return an ``LSTMSpamModel`` instance (not yet trained)."""
    return LSTMSpamModel(
        embedding_dim=embedding_dim,
        lstm1_units=lstm1_units,
        lstm2_units=lstm2_units,
        dropout_rate=dropout_rate,
        dense_hidden_units=dense_hidden_units,
    )


__all__ = [
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_MAX_LEN",
    "LSTMSpamModel",
    "build_lstm_spam_model",
]
