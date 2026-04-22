"""
Keras model blueprint: stacked LSTM + dense head for binary spam/ham (Word2Vec sequence input).
"""
from __future__ import annotations

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM

# Defaults aligned with the Word2Vec model (100-dim) and LSTM input padding
DEFAULT_MAX_LEN = 50
DEFAULT_EMBEDDING_DIM = 100


def build_lstm_spam_model(
    max_len: int = DEFAULT_MAX_LEN,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    *,
    lstm1_units: int = 64,
    lstm2_units: int = 32,
    dropout_rate: float = 0.3,
    dense_hidden_units: int = 32,
) -> Sequential:
    """
    Return an uncompiled ``Sequential`` model.

    Input shape: ``(max_len, embedding_dim)`` — one W2V vector per timestep.
    """
    return Sequential(
        [
            Input(shape=(max_len, embedding_dim)),
            LSTM(lstm1_units, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(lstm2_units),
            Dropout(dropout_rate),
            Dense(dense_hidden_units, activation="relu"),
            Dense(1, activation="sigmoid"),
        ],
        name="lstm_spam_w2v",
    )


__all__ = [
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_MAX_LEN",
    "build_lstm_spam_model",
]
