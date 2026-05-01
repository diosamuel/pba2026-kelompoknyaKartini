from __future__ import annotations

import os
import re
import sys

import torch

_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from deep_learning.dataloader import full_preprocess
from new_deeplearning.model import SpamClassifier


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", str(text).lower())


def encode(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    tokens = tokenize(text)
    ids = [vocab.get(t, 1) for t in tokens]  # 1 = <UNK>
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))  # 0 = <PAD>
    else:
        ids = ids[:max_len]
    return ids


def load_trained_model(model_path: str):
    checkpoint = torch.load(model_path, map_location="cpu")
    vocab: dict[str, int] = checkpoint["vocab"]
    max_len: int = checkpoint.get("max_len", 50)

    model = SpamClassifier(vocab_size=len(vocab))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocab, max_len


def predict_text(model, vocab: dict[str, int], max_len: int, raw_text: str):
    # Training uses `df["clean"]`, so apply the same preprocessing chain for inference.
    clean_text = full_preprocess(raw_text)
    x_ids = encode(clean_text, vocab=vocab, max_len=max_len)
    x = torch.tensor([x_ids], dtype=torch.long)
    with torch.no_grad():
        prob = float(model(x).item())
    label = "spam" if prob >= 0.5 else "ham"
    return label, prob, clean_text


def main() -> None:
    model_path = os.path.join(_ROOT, "new_deeplearning", "model", "spam_classifier_lstm.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run `python -m new_deeplearning.main` first."
        )

    model, vocab, max_len = load_trained_model(model_path)
    print(f"Loaded model: {model_path}")
    print("Type an email text. Enter 'exit' to stop.")

    while True:
        text = input("\nInput text: ").strip()
        if text.lower() in {"exit", "quit", "q"}:
            print("Done.")
            break
        if not text:
            print("Please type some text.")
            continue

        label, prob, clean_text = predict_text(model, vocab, max_len, text)
        print(f"Predicted label : {label}")
        print(f"Spam probability: {prob:.4f}")
        print(f"Clean text      : {clean_text}")


if __name__ == "__main__":
    main()
