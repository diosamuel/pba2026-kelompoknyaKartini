from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from deep_learning.model import SpamClassifier


# set up the base model
def build_training_objects(vocab):
    vocab_size = len(vocab)
    model = SpamClassifier(vocab_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, criterion, optimizer, device


# train epoch
def train_epochs(model, criterion, optimizer, device, train_dl, epochs=30):
    train_losses = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg = train_loss / len(train_dl)
        train_losses.append(avg)
        print(f"Epoch {epoch + 1} | Loss: {avg:.4f}")
    return train_losses
