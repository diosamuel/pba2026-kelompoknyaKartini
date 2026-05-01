from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


# turn text to code vector
def encode(text, tokenize, vocab, max_len=50):
    tokens = tokenize(text)
    ids = [vocab.get(t, 1) for t in tokens]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids


# set up placeholder Dataset
class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenize, vocab, max_len=50):
        self.texts = [
            torch.tensor(encode(t, tokenize=tokenize, vocab=vocab, max_len=max_len), dtype=torch.long)
            for t in texts
        ]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


# set up the dataloader
def build_dataloaders(X_train, y_train, X_test, y_test, tokenize, vocab, batch_size=32, max_len=50):
    label_map = {"ham": 0, "spam": 1}

    train_labels = y_train.map(label_map).to_numpy(dtype="float32")
    test_labels = y_test.map(label_map).to_numpy(dtype="float32")

    train_texts = X_train
    test_texts = X_test

    train_ds = SpamDataset(train_texts, train_labels, tokenize=tokenize, vocab=vocab, max_len=max_len)
    test_ds = SpamDataset(test_texts, test_labels, tokenize=tokenize, vocab=vocab, max_len=max_len)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size)
    return train_dl, test_dl
