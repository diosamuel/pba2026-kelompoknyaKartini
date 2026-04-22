"""
ML helpers layered on :mod:`text_preprocess`: Word2Vec document vectors, small SVM train/predict.

- **Preprocessing** (cleaning, normalizing, stopwords) comes from ``text_preprocess``; this file only
  adds **embedding** and **svm** on top of ``full_preprocess_tokens``.

The main Streamlit app (``app.py``) uses PyCaret models in ``machine_learning/``; ``index.py`` uses
this module to train a lightweight SVM in-process for demos.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.svm import SVC

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from text_preprocess import full_preprocess_tokens

_DEFAULT_DATASET = os.path.join(_REPO_ROOT, "dataset", "email_spam_indo.csv")


def full_pipeline(text: str) -> list[str]:
    """Token list for ML; alias for :func:`text_preprocess.full_preprocess_tokens`."""
    return full_preprocess_tokens(text)


def document_vector(tokens: list[str], model: Word2Vec) -> np.ndarray:
    valid = [w for w in tokens if w in model.wv.key_to_index]
    if not valid:
        return np.zeros(model.vector_size)
    return np.mean(model.wv[valid], axis=0)


def train_model() -> tuple[Word2Vec, SVC]:
    df = pd.read_csv(_DEFAULT_DATASET)
    df = df.rename(columns={"Pesan": "text", "Kategori": "label"})

    df["tokens"] = df["text"].map(full_pipeline)

    w2v = Word2Vec(
        sentences=df["tokens"],
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
    )

    X = np.array([document_vector(t, w2v) for t in df["tokens"]])
    y = (df["label"] == "spam").astype(int).values

    svm = SVC(kernel="linear", probability=True, random_state=42)
    svm.fit(X, y)

    return w2v, svm


def predict_spam(text: str, w2v: Word2Vec, model: SVC) -> tuple[str, float]:
    tokens = full_pipeline(text)
    vec = document_vector(tokens, w2v).reshape(1, -1)
    proba = model.predict_proba(vec)[0]
    pred = model.predict(vec)[0]
    label = "SPAM" if pred == 1 else "HAM (Bukan Spam)"
    confidence = float(proba[pred])
    return label, confidence
