from __future__ import annotations

import os
import re
import sys
from collections import Counter

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from deep_learning.dataloader import build_dataloaders
from deep_learning.train_lstm import build_training_objects, train_epochs


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", str(text).lower())


def build_vocab(texts, min_freq: int = 2) -> dict[str, int]:
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))

    # 0 = padding, 1 = unknown
    vocab: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for token, freq in counter.items():
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def main() -> None:
    print("Loading dataset...")
    dataset_path = os.path.join(_ROOT, "dataset", "email_spam_indo.csv")
    df = pd.read_csv(dataset_path, encoding="utf-8", on_bad_lines="skip")

    required_columns = {"Pesan", "Kategori"}
    if not required_columns.issubset(df.columns):
        raise ValueError("Dataset must include 'Pesan' and 'Kategori' columns.")

    data = (
        df[["Pesan", "Kategori"]]
        .dropna()
        .rename(columns={"Pesan": "text", "Kategori": "label"})
    )
    data["text"] = data["text"].astype(str).str.strip()
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data = data[(data["text"] != "") & (data["label"].isin(["spam", "ham"]))]

    X = data["text"]
    y = data["label"]

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Building vocabulary...")
    vocab = build_vocab(X_train, min_freq=2)
    print(f"Vocab size: {len(vocab)}")

    print("Building dataloaders...")
    train_dl, test_dl = build_dataloaders(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        tokenize=tokenize,
        vocab=vocab,
        batch_size=32,
        max_len=50,
    )
    print(f"Train batches: {len(train_dl)} | Test batches: {len(test_dl)}")

    print("Initializing model and optimizer...")
    model, criterion, optimizer, device = build_training_objects(vocab=vocab)
    print(f"Using device: {device}")

    print("Training...")
    train_epochs(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_dl=train_dl,
        epochs=30,
    )

    model_dir = os.path.join(_ROOT, "deep_learning", "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "spam_classifier_lstm.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "max_len": 50,
        },
        model_path,
    )
    print(f"Saved model to: {model_path}")

    print("Training finished.")


if __name__ == "__main__":
    main()
