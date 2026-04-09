import re

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.svm import SVC


NORMALIZATION_DICT = {
    "gk": "tidak", "d": "di", "ga": "tidak", "gak": "tidak",
    "nggak": "tidak", "tdk": "tidak", "dr": "dari", "dgn": "dengan",
    "utk": "untuk", "yg": "yang", "jg": "juga", "krn": "karena",
    "blm": "belum", "sdh": "sudah", "aja": "saja", "trs": "terus",
    "bgt": "banget", "dlm": "dalam",
    "rp": "rupiah", "jt": "juta", "rb": "ribu",
    "cashback": "cashback", "promo": "promosi", "diskon": "diskon",
    "gratis": "gratis", "hadiah": "hadiah", "menang": "menang",
    "undian": "undian",
    "tlp": "telepon", "telp": "telepon", "hp": "handphone",
    "no": "nomor", "rek": "rekening", "an": "atas nama",
    "selamat!!!": "selamat", "selamat!!": "selamat", "selamat!": "selamat",
    "trnsfer": "transfer", "trf": "transfer",
    "4nda": "anda", "4pa": "apa", "1ni": "ini", "k4mu": "kamu",
    "klik": "klik", "segera": "segera", "buruan": "cepat",
    "cepat": "cepat", "terbatas": "terbatas", "khusus": "khusus",
    "free": "gratis", "win": "menang", "winner": "pemenang",
    "prize": "hadiah", "bonus": "bonus", "gift": "hadiah", "reward": "hadiah",
    "cash": "uang", "money": "uang", "credit": "kredit",
    "loan": "pinjaman", "bank": "bank", "transfer": "transfer",
    "account": "akun", "sspecial": "spesial", "balance": "saldo",
    "promo": "promosi", "offer": "penawaran", "deal": "penawaran",
    "discount": "diskon", "sale": "diskon", "limited": "terbatas",
    "exclusive": "eksklusif",
    "click": "klik", "claim": "klaim", "buy": "beli", "order": "pesan",
    "register": "daftar", "subscribe": "langganan", "join": "gabung",
    "verify": "verifikasi", "confirm": "konfirmasi",
    "urgent": "segera", "now": "sekarang", "today": "hari ini",
    "instant": "instan",
    "software": "perangkat lunak", "system": "sistem",
    "update": "pembaruan", "download": "unduh", "install": "instal",
    "hello": "halo", "dear": "halo", "sir": "bapak", "madam": "ibu",
    "service": "layanan", "customer": "pelanggan", "support": "dukungan",
    "information": "informasi", "message": "pesan",
}

_stopword_factory = StopWordRemoverFactory()
_stopwords = set(_stopword_factory.get_stop_words())


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(
        r'(dikeret oleh|ditahan oleh|ect pada|subjek:).*', '',
        text, flags=re.IGNORECASE,
    )
    text = re.sub(r'\b\w+\s+(com|net|org|co|id)\b', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'^(re|fw|fwd):[^.?!\n]*', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize(text: str) -> str:
    return ' '.join(NORMALIZATION_DICT.get(w, w) for w in text.split())


def remove_stopwords(text: str) -> str:
    return ' '.join(w for w in text.split() if w not in _stopwords)


def tokenize(text: str) -> list[str]:
    return text.split()


def full_pipeline(text: str) -> list[str]:
    text = preprocess(text)
    text = normalize(text)
    text = remove_stopwords(text)
    return tokenize(text)


def document_vector(tokens: list[str], model: Word2Vec) -> np.ndarray:
    valid = [w for w in tokens if w in model.wv.key_to_index]
    if not valid:
        return np.zeros(model.vector_size)
    return np.mean(model.wv[valid], axis=0)


def train_model() -> tuple[Word2Vec, SVC]:
    df = pd.read_csv('../dataset/email_spam_indo.csv')
    df = df.rename(columns={'Pesan': 'text', 'Kategori': 'label'})

    df['tokens'] = df['text'].apply(full_pipeline)

    w2v = Word2Vec(
        sentences=df['tokens'],
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
    )

    X = np.array([document_vector(t, w2v) for t in df['tokens']])
    y = (df['label'] == 'spam').astype(int).values

    svm = SVC(kernel='linear', probability=True, random_state=42)
    svm.fit(X, y)

    return w2v, svm


def predict_spam(text: str, w2v: Word2Vec, model: SVC) -> tuple[str, float]:
    tokens = full_pipeline(text)
    vec = document_vector(tokens, w2v).reshape(1, -1)
    proba = model.predict_proba(vec)[0]
    pred = model.predict(vec)[0]
    label = "SPAM" if pred == 1 else "HAM (Bukan Spam)"
    confidence = proba[pred]
    return label, confidence
