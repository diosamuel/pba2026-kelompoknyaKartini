"""
Load Indonesian spam email datasets and run the same preprocessing as inference (app).

Training scripts and ``src.program.app`` use this module so data handling stays in one place.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import pandas as pd

# Repository root = parent of this package (``deep_learning/`` → project root)
_DL_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_DL_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from text_preprocess import (  # noqa: E402
    NORMALIZATION_DICT,
    full_preprocess,
    full_preprocess_tokens,
    normalize_text,
    preprocess,
    preprocess_stages,
    remove_stopwords,
)

# --- Paths (all resolved from repository root) ---

# Default CSV: Indonesian columns Pesan + Kategori (see ``load_spam_email_dataframe``)
DEFAULT_SPAM_CSV = os.path.join(_REPO_ROOT, "dataset", "email_spam_indo.csv")
# Word2vec + PyCaret artifacts live with classical ML
DEFAULT_W2V_PATH = os.path.join(_REPO_ROOT, "machine_learning", "w2v_kamus.model")

RAW_TEXT_COL = "Pesan"
RAW_LABEL_COL = "Kategori"
STANDARD_TEXT_COL = "text"
STANDARD_LABEL_COL = "label"
CLEAN_COL = "clean"


def default_spam_csv_path() -> str:
    return DEFAULT_SPAM_CSV


def default_w2v_path() -> str:
    return DEFAULT_W2V_PATH


def add_clean_column(
    df: pd.DataFrame,
    text_col: str = STANDARD_TEXT_COL,
    clean_col: str = CLEAN_COL,
) -> pd.DataFrame:
    """Apply full_preprocess to each row in *text_col*; store result in *clean_col*."""
    out = df.copy()
    out[clean_col] = out[text_col].map(full_preprocess)
    return out


def load_spam_email_dataframe(
    csv_path: Optional[str] = None,
    *,
    add_clean: bool = True,
) -> pd.DataFrame:
    """
    Load the spam/ham CSV and standardize column names; optionally add a ``clean`` column.

    Default path: ``<repo>/dataset/email_spam_indo.csv``.
    """
    path = os.path.normpath(csv_path or DEFAULT_SPAM_CSV)
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            RAW_TEXT_COL: STANDARD_TEXT_COL,
            RAW_LABEL_COL: STANDARD_LABEL_COL,
        }
    )
    if add_clean:
        df = add_clean_column(df, text_col=STANDARD_TEXT_COL, clean_col=CLEAN_COL)
    return df


__all__ = [
    "CLEAN_COL",
    "DEFAULT_SPAM_CSV",
    "DEFAULT_W2V_PATH",
    "NORMALIZATION_DICT",
    "RAW_LABEL_COL",
    "RAW_TEXT_COL",
    "STANDARD_LABEL_COL",
    "STANDARD_TEXT_COL",
    "add_clean_column",
    "default_spam_csv_path",
    "default_w2v_path",
    "full_preprocess",
    "full_preprocess_tokens",
    "load_spam_email_dataframe",
    "normalize_text",
    "preprocess",
    "preprocess_stages",
    "remove_stopwords",
]
