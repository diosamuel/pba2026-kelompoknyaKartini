"""
Shared Indonesian email text preprocessing for training (LSTM) and inference (Streamlit app).
"""
from __future__ import annotations

import re
from typing import Tuple

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# --- Cleaning (lower, strip URLs, email noise, keep letters/spaces) ---

def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(
        r"(dikeret oleh|ditahan oleh|ect pada|subjek:).*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b\w+\s+(com|net|org|co|id)\b", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"^(re|fw|fwd):[^.?!\n]*", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --- Normalization (colloquial / spam variants → canonical tokens) ---

NORMALIZATION_DICT = {
    "gk": "tidak",
    "ga": "tidak",
    "gak": "tidak",
    "nggak": "tidak",
    "tdk": "tidak",
    "dr": "dari",
    "dgn": "dengan",
    "utk": "untuk",
    "yg": "yang",
    "jg": "juga",
    "krn": "karena",
    "blm": "belum",
    "sdh": "sudah",
    "aja": "saja",
    "trs": "terus",
    "bgt": "banget",
    "dlm": "dalam",
    "rp": "rupiah",
    "jt": "juta",
    "rb": "ribu",
    "cashback": "cashback",
    "promo": "promosi",
    "diskon": "diskon",
    "gratis": "gratis",
    "hadiah": "hadiah",
    "menang": "menang",
    "undian": "undian",
    "tlp": "telepon",
    "telp": "telepon",
    "hp": "handphone",
    "no": "nomor",
    "rek": "rekening",
    "selamat!!!": "selamat",
    "selamat!!": "selamat",
    "selamat!": "selamat",
    "trnsfer": "transfer",
    "trf": "transfer",
    "4nda": "anda",
    "4pa": "apa",
    "1ni": "ini",
    "k4mu": "kamu",
    "klik": "klik",
    "segera": "segera",
    "buruan": "cepat",
    "cepat": "cepat",
    "terbatas": "terbatas",
    "khusus": "khusus",
    "free": "gratis",
    "win": "menang",
    "winner": "pemenang",
    "prize": "hadiah",
    "bonus": "bonus",
    "gift": "hadiah",
    "reward": "hadiah",
    "cash": "uang",
    "money": "uang",
    "credit": "kredit",
    "loan": "pinjaman",
    "bank": "bank",
    "transfer": "transfer",
    "account": "akun",
    "sspecial": "spesial",
    "balance": "saldo",
    "offer": "penawaran",
    "deal": "penawaran",
    "discount": "diskon",
    "sale": "diskon",
    "limited": "terbatas",
    "exclusive": "eksklusif",
    "click": "klik",
    "claim": "klaim",
    "buy": "beli",
    "order": "pesan",
    "register": "daftar",
    "subscribe": "langganan",
    "join": "gabung",
    "verify": "verifikasi",
    "confirm": "konfirmasi",
    "urgent": "segera",
    "now": "sekarang",
    "today": "hari ini",
    "instant": "instan",
    "software": "perangkat lunak",
    "system": "sistem",
    "update": "pembaruan",
    "download": "unduh",
    "install": "instal",
    "hello": "halo",
    "dear": "halo",
    "sir": "bapak",
    "madam": "ibu",
    "service": "layanan",
    "customer": "pelanggan",
    "support": "dukungan",
    "information": "informasi",
    "message": "pesan",
}


def normalize_text(text: str) -> str:
    words = text.split()
    normalized_words = [NORMALIZATION_DICT.get(word, word) for word in words]
    return " ".join(normalized_words)


# --- Stopwords (Sastrawi) ---

_factory = StopWordRemoverFactory()
_stopwords = set(_factory.get_stop_words())


def remove_stopwords(text: str) -> str:
    return " ".join([word for word in text.split() if word not in _stopwords])


def full_preprocess(text: str) -> str:
    return remove_stopwords(normalize_text(preprocess(text)))


def full_preprocess_tokens(text: str) -> list[str]:
    """Tokens after the full clean → normalize → stopword pipeline (for W2V / ML)."""
    return full_preprocess(text).split()


def preprocess_stages(text: str) -> Tuple[str, str, str]:
    """
    Run cleaning → normalization → stopword removal.
    Returns (after_preprocess, after_normalize, after_stopwords) for UIs that show each step.
    """
    teks_bersih = preprocess(text)
    teks_normal = normalize_text(teks_bersih)
    teks_stopword = remove_stopwords(teks_normal)
    return teks_bersih, teks_normal, teks_stopword
