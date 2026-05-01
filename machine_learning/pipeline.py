from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from config import AppConfig
from text_preprocess import SpamHamDatasetCleaner


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        self._w2v_model = None
        self._vector_size: int | None = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z]+", str(text).lower())

    def _load_or_train_model(self, texts: pd.Series) -> None:
        try:
            from gensim.models import Word2Vec
        except ImportError as exc:
            raise ImportError(
                "gensim is required for Word2Vec features. Install it with `pip install gensim`."
            ) from exc

        if self.model_path.exists():
            self._w2v_model = Word2Vec.load(str(self.model_path))
            self._vector_size = int(self._w2v_model.vector_size)
            return

        sentences = [self._tokenize(text) for text in texts]
        sentences = [tokens for tokens in sentences if tokens]
        if not sentences:
            raise ValueError("Cannot train Word2Vec because no tokens were found.")

        self._w2v_model = Word2Vec(
            sentences=sentences,
            vector_size=100,
            window=5,
            min_count=1,
            workers=1,
            seed=42,
        )
        self._vector_size = int(self._w2v_model.vector_size)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self._w2v_model.save(str(self.model_path))
        print(f"Trained and saved Word2Vec model to {self.model_path}")

    def fit(self, X: pd.Series, y=None) -> "Word2VecVectorizer":
        self._load_or_train_model(pd.Series(X))
        return self

    def transform(self, X: pd.Series) -> np.ndarray:
        if self._w2v_model is None or self._vector_size is None:
            raise RuntimeError("Word2VecVectorizer must be fitted before transform.")

        vectors: list[np.ndarray] = []
        for text in pd.Series(X):
            tokens = self._tokenize(text)
            token_vectors = [
                self._w2v_model.wv[token]
                for token in tokens
                if token in self._w2v_model.wv
            ]
            if token_vectors:
                vectors.append(np.mean(token_vectors, axis=0))
            else:
                vectors.append(np.zeros(self._vector_size, dtype=np.float32))
        return np.vstack(vectors)


class DatasetLoader:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def load(self) -> pd.DataFrame:
        cleaner = SpamHamDatasetCleaner(
            inputPath=str(self.config.dataset_path),
            labelColumn=self.config.label_column,
            textColumn=self.config.text_column,
            includeOriginalMessage=False,
        )
        cleaned_df = cleaner.run()

        data = cleaned_df[[self.config.text_column, self.config.label_column]].rename(
            columns={
                self.config.text_column: "text",
                self.config.label_column: "label",
            }
        )
        data["text"] = data["text"].astype(str).str.strip()
        data["label"] = data["label"].astype(str).str.strip().str.lower()
        data = data[
            (data["text"] != "")
            & (data["label"].isin(self.config.allowed_labels))
        ]
        return data.reset_index(drop=True).copy()
