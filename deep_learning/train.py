"""
Train the LSTM spam detector using :mod:`deep_learning.dataloader` and
:func:`deep_learning.model.build_lstm_spam_model`, with Weights & Biases logging.

Framework: PyTorch
"""
from __future__ import annotations

import os
import sys
from contextlib import suppress

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import wandb
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from deep_learning.dataloader import default_w2v_path, load_spam_email_dataframe  # noqa: E402
from deep_learning.model import (  # noqa: E402
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MAX_LEN,
    build_lstm_spam_model,
)

_DL_DIR = os.path.dirname(os.path.abspath(__file__))


def get_sequence_vectors(
    text: str,
    w2v_model: Word2Vec,
    max_len: int = DEFAULT_MAX_LEN,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> np.ndarray:
    words = text.split()
    vector_seq = np.zeros((max_len, embedding_dim))
    for i, word in enumerate(words):
        if i >= max_len:
            break
        if word in w2v_model.wv.key_to_index:
            vector_seq[i] = w2v_model.wv[word]
    return vector_seq


def _default_train_config() -> dict:
    """Edit these (or a W&B sweep) to experiment. Logged 1:1 to wandb.config."""
    return {
        "entity": "koding12sam-itera",
        "project": "email-spam-detection-lstm",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 30,
        "test_size": 0.2,
        "random_state": 42,
        "dataset": "email_spam_indo",
        "max_len": DEFAULT_MAX_LEN,
        "embedding_dim": DEFAULT_EMBEDDING_DIM,
        "architecture": "stacked_LSTM_w2v",
        "lstm1_units": 64,
        "lstm2_units": 32,
        "dropout_rate": 0.3,
        "dense_hidden_units": 32,
    }


def main() -> None:
    cfg_dict = _default_train_config()
    entity = cfg_dict.pop("entity", None)
    project = cfg_dict.pop("project", "email-spam-detection-lstm")

    run = wandb.init(
        entity=entity,
        project=project,
        config=cfg_dict,
    )
    c = run.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")

    try:
        print("Memuat dataset dan preprocessing...")
        df = load_spam_email_dataframe()

        print("Memuat Word2Vec model...")
        w2v_model = Word2Vec.load(default_w2v_path())

        print("Membuat urutan vektor (sequences)...")
        X = np.array(
            [
                get_sequence_vectors(
                    t,
                    w2v_model,
                    max_len=c.max_len,
                    embedding_dim=c.embedding_dim,
                )
                for t in df["clean"]
            ]
        )

        le = LabelEncoder()
        y = le.fit_transform(df["label"])

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=c.test_size,
            random_state=c.random_state,
            stratify=y,
        )

        run.config.update(
            {
                "n_total": int(len(X)),
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "n_features_time": int(c.max_len),
                "n_features_dim": int(c.embedding_dim),
            },
            allow_val_change=True,
        )

        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

        # Convert to PyTorch tensors
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
        X_test_t = torch.FloatTensor(X_test).to(device)
        y_test_t = torch.FloatTensor(y_test).unsqueeze(1).to(device)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True)

        print("Membangun model LSTM (PyTorch)...")
        model = build_lstm_spam_model(
            embedding_dim=c.embedding_dim,
            lstm1_units=c.lstm1_units,
            lstm2_units=c.lstm2_units,
            dropout_rate=c.dropout_rate,
            dense_hidden_units=c.dense_hidden_units,
        ).to(device)

        print(model)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=c.learning_rate)

        print("Melatih model LSTM...")
        for epoch in range(c.epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_X.size(0)
                predicted = (outputs > 0.5).float()
                train_correct += (predicted == batch_y).sum().item()
                train_total += batch_y.size(0)

            train_loss /= train_total
            train_acc = train_correct / train_total

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_t)
                val_loss = criterion(val_outputs, y_test_t).item()
                val_predicted = (val_outputs > 0.5).float()
                val_acc = (val_predicted == y_test_t).sum().item() / len(y_test_t)

            print(
                f"Epoch {epoch + 1}/{c.epochs} - "
                f"loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - "
                f"val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}"
            )

            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
            })

        run.summary["val_accuracy_last"] = val_acc

        print("Mengevaluasi model pada test data...")
        model.eval()
        with torch.no_grad():
            y_pred_probs = model(X_test_t).cpu().numpy()
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()

        test_acc = float(accuracy_score(y_test, y_pred))
        test_f1 = float(f1_score(y_test, y_pred, average="binary", zero_division=0))
        run.log(
            {
                "test/accuracy": test_acc,
                "test/f1": test_f1,
            }
        )
        print(f"Accuracy: {test_acc}")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        report_path = os.path.join(_DL_DIR, "classification_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(
                classification_report(y_test, y_pred, target_names=le.classes_)
            )
        with suppress(Exception):
            rep = wandb.Artifact("classification_report", type="result")
            rep.add_file(report_path, name="classification_report.txt")
            run.log_artifact(rep)

        model_path = os.path.join(_DL_DIR, "model", "spam_model_lstm.pth")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model LSTM berhasil disimpan ke {model_path}!")
        with suppress(Exception):
            m_art = wandb.Artifact("lstm-pytorch", type="model")
            m_art.add_file(model_path, name="spam_model_lstm.pth")
            run.log_artifact(m_art)

        run.summary["label_0"] = str(le.classes_[0])
        run.summary["label_1"] = str(le.classes_[1])

        print(f"Label mapping: 0 -> {le.classes_[0]}, 1 -> {le.classes_[1]}")
    finally:
        wandb.finish()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
