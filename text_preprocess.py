"""
Clean ``dataset/email_spam_indo.csv`` using preprocessing classes defined in this file.

This script is Colab-friendly:
- ``inputPath`` is a plain string path.
- ``run()`` returns a DataFrame (no file write by default).

Run from notebook/cell:
    cleaner = SpamHamDatasetCleaner(inputPath="/content/email_spam_indo.csv")
    dfClean = cleaner.run()
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

class EmailBodyPreprocessor:
    """
    Noise normalization, structural scrub, and Indonesian stopword removal for email body text.
    """

    def __init__(self) -> None:
        self._stopwords = frozenset(StopWordRemoverFactory().get_stop_words())

    def scrubStructural(self, text: str) -> str:
        """Collapse long dash separator lines and excessive blank lines."""
        t = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
        t = re.sub(r"(?m)^[-_=]{10,}\s*$", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t.strip()

    def normalizeNoise(self, text: str) -> str:
        """Lowercase, drop URLs and emails, trim common header fragments, letters and spaces only."""
        t = str(text).lower()
        t = re.sub(r"https?://\S+|www\.\S+", " ", t)
        t = re.sub(r"http\S+", " ", t)
        t = re.sub(
            r"(dikeret oleh|ditahan oleh|diteruskan oleh|forwarded by|ect pada|subjek:).*",
            " ",
            t,
            flags=re.IGNORECASE,
        )
        t = re.sub(r"\b\w+\s+(com|net|org|co|id)\b", " ", t)
        t = re.sub(r"\S+@\S+", " ", t)
        t = re.sub(r"^(re|fw|fwd):[^.?!\n]*", " ", t, flags=re.IGNORECASE)
        t = re.sub(r"[^a-zA-Z\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def removeStopwords(self, text: str) -> str:
        """Remove Sastrawi Indonesian stopwords."""
        return " ".join(w for w in text.split() if w not in self._stopwords)

    def fullClean(self, text: str) -> str:
        """Structural scrub → noise normalization → stopword removal."""
        t = self.scrubStructural(text)
        t = self.normalizeNoise(t)
        return self.removeStopwords(t)


@dataclass
class SpamHamDatasetCleaner:
    """Load CSV, validate rows, run :class:`EmailBodyPreprocessor`, return cleaned DataFrame."""

    inputPath: str
    minCleanLength: int = 15
    dedupeOnCleanText: bool = True
    includeOriginalMessage: bool = False
    labelColumn: str = "Kategori"
    textColumn: str = "Pesan"
    allowedLabels: frozenset[str] = field(default_factory=lambda: frozenset({"spam", "ham"}))
    stats: dict[str, int] = field(default_factory=dict)
    preprocessor: EmailBodyPreprocessor = field(default_factory=EmailBodyPreprocessor)

    def normalizeLabel(self, value: object) -> str | None:
        if pd.isna(value):
            return None
        x = str(value).strip().lower()
        if x in self.allowedLabels:
            return x
        return None

    def loadCsv(self) -> pd.DataFrame:
        return pd.read_csv(self.inputPath, encoding="utf-8", on_bad_lines="skip")

    def buildCleanFrame(self, df: pd.DataFrame) -> pd.DataFrame:
        present = set(df.columns)
        if self.textColumn not in present or self.labelColumn not in present:
            raise ValueError(
                f"The CSV must include columns {self.labelColumn!r} and {self.textColumn!r}."
            )

        self.stats = {"rowsRead": len(df)}
        d = df[[self.labelColumn, self.textColumn]].copy()
        d = d.rename(columns={self.labelColumn: "label", self.textColumn: "text_raw"})

        d = d.dropna(subset=["text_raw"])
        d["text_raw"] = d["text_raw"].astype(str).str.strip()
        d = d.loc[d["text_raw"].map(len) > 0]
        self.stats["afterDropEmptyText"] = len(d)

        d["label"] = d["label"].map(self.normalizeLabel)
        d = d.dropna(subset=["label"])
        self.stats["afterValidLabel"] = len(d)

        d["clean"] = d["text_raw"].map(self.preprocessor.fullClean)

        d = d.loc[d["clean"].str.len() >= self.minCleanLength]
        self.stats["afterMinLength"] = len(d)

        if self.dedupeOnCleanText:
            d = d.drop_duplicates(subset=["clean"], keep="first")
            self.stats["afterDedupeClean"] = len(d)

        return d

    def buildOutputFrame(self, d: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame({"Kategori": d["label"], "Pesan": d["clean"]})
        if self.includeOriginalMessage:
            out["Pesan_asli"] = d["text_raw"].values
        return out

    def printSummary(self, out: pd.DataFrame) -> None:
        print("Spam versus ham dataset cleaning summary.")
        n = self.stats.get("rowsRead")
        if n is not None:
            print(f"Rows read from the source file: {n}.")

        n = self.stats.get("afterDropEmptyText")
        if n is not None:
            print(f"After removing empty message text: {n} rows.")

        n = self.stats.get("afterValidLabel")
        if n is not None:
            print(f"After keeping only spam and ham labels: {n} rows.")

        n = self.stats.get("afterMinLength")
        if n is not None:
            print(f"After minimum clean-text length filter: {n} rows.")

        n = self.stats.get("afterDedupeClean")
        if n is not None:
            print(f"After deduplication on clean text: {n} rows.")

        print(f"Total rows in cleaned DataFrame: {len(out)}.")

        print("Count by class.")
        for lab, count in out["Kategori"].value_counts().items():
            print(f"  Class {lab}: {count} examples.")

        print("Cleaned DataFrame is ready for model input.")

    def run(self) -> pd.DataFrame:
        raw = self.loadCsv()
        cleaned = self.buildCleanFrame(raw)
        out = self.buildOutputFrame(cleaned)
        self.printSummary(out)
        return out


cleaner = SpamHamDatasetCleaner(inputPath="/content/email_spam_indo.csv")
dfClean = cleaner.run()