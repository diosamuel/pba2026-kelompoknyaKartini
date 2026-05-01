"""
Train classical classifiers (SVM, Naive Bayes, Logistic Regression) for Indonesian spam email.

Data schema (``dataset/email_spam_indo.csv``)
----------------------------------------------
- **Kategori**: label, values such as ``spam`` / ``ham`` (loaded as ``text`` / ``label`` via
  :func:`deep_learning.dataloader.load_spam_email_dataframe`).
- **Pesan**: raw email body text.

Pipeline
--------
#. Load CSV via ``deep_learning.dataloader`` (same cleaning as LSTM + Streamlit app).
#. Train Word2Vec on tokenized ``clean`` text (100-dim, aligned with ``deep_learning.model.DEFAULT_EMBEDDING_DIM``).
#. Mean-pool word vectors → feature rows ``w2v_0`` … ``w2v_99``.
#. PyCaret fits SVM, NB, LR; artifacts written to ``machine_learning/model/`` and W2V to
   ``machine_learning/w2v_kamus.model``.

**Note:** :mod:`deep_learning.model` only defines the **PyTorch LSTM**; PyCaret ``.pkl`` pipelines live
under ``machine_learning/model/`` (see :mod:`src.program.app`).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from pycaret.classification import create_model, save_model, setup

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from deep_learning.dataloader import (  # noqa: E402
    CLEAN_COL,
    STANDARD_LABEL_COL,
    load_spam_email_dataframe,
)
from deep_learning.model import DEFAULT_EMBEDDING_DIM  # noqa: E402

# Match Streamlit / training convention
_ML_DIR = os.path.join(_REPO_ROOT, "machine_learning")
_MODEL_DIR = os.path.join(_ML_DIR, "model")
_W2V_PATH = os.path.join(_ML_DIR, "w2v_kamus.model")

_W2V_PARAMS = {
    "vector_size": DEFAULT_EMBEDDING_DIM,
    "window": 5,
    "min_count": 2,
    "workers": 4,
    "seed": 42,
}


def _document_vector(tokens: list[str], model: Word2Vec) -> np.ndarray:
    valid = [w for w in tokens if w in model.wv.key_to_index]
    if not valid:
        return np.zeros(model.vector_size)
    return np.mean(model.wv[valid], axis=0)


def build_feature_frame(df: pd.DataFrame, w2v: Word2Vec) -> pd.DataFrame:
    sentences = [str(s).split() for s in df[CLEAN_COL]]
    X = np.array([_document_vector(s, w2v) for s in sentences])
    out = pd.DataFrame(X, columns=[f"w2v_{i}" for i in range(w2v.vector_size)])
    out[STANDARD_LABEL_COL] = df[STANDARD_LABEL_COL].astype(str).str.lower().values
    return out.reset_index(drop=True)


def main() -> None:
    print("Memuat dataset (Kategori, Pesan) + preprocessing...")
    df = load_spam_email_dataframe()
    df[STANDARD_LABEL_COL] = df[STANDARD_LABEL_COL].astype(str).str.lower()

    print("Melatih Word2Vec pada kolom clean...")
    sentences = [str(s).split() for s in df[CLEAN_COL]]
    w2v = Word2Vec(sentences=sentences, **_W2V_PARAMS)
    os.makedirs(_ML_DIR, exist_ok=True)
    w2v.save(_W2V_PATH)
    print(f"Word2Vec disimpan: {_W2V_PATH}")

    features = build_feature_frame(df, w2v)
    print(f"Fitur: {features.shape[0]} baris, {features.shape[1] - 1} dim + label")

    os.makedirs(_MODEL_DIR, exist_ok=True)

    setup(
        data=features,
        target=STANDARD_LABEL_COL,
        session_id=42,
        verbose=False,
        html=False,
    )

    to_train = (
        ("spam_model_svm", "svm"),
        ("spam_model_nb", "nb"),
        ("spam_model_lr", "lr"),
    )

    for save_stem, model_id in to_train:
        print(f"Melatih & menyimpan PyCaret model: {model_id} → {save_stem}...")
        m = create_model(model_id)
        out_base = os.path.join(_MODEL_DIR, save_stem)
        save_model(m, out_base)
        print(f"  OK: {out_base} (+ .pkl)")

    print("Selesai. Gunakan artefak ini di src/program/app.py (load_model path tanpa .pkl).")


if __name__ == "__main__":
    main()
