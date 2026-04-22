import pandas as pd
import numpy as np
import re
import os
from gensim.models import Word2Vec
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input

print("Memuat dataset...")
df = pd.read_csv('../dataset/email_spam_indo.csv')
df = df.rename(columns={'Pesan': 'text', 'Kategori': 'label'})

print("Memulai preprocessing...")

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'(dikeret oleh|ditahan oleh|ect pada|subjek:).*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\w+\s+(com|net|org|co|id)\b', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'^(re|fw|fwd):[^.?!\n]*', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

NORMALIZATION_DICT = {
    "gk": "tidak", "d": "di", "ga": "tidak", "gak": "tidak", "nggak": "tidak", "tdk": "tidak",
    "dr": "dari", "dgn": "dengan", "utk": "untuk", "yg": "yang", "jg": "juga", "krn": "karena",
    "blm": "belum", "sdh": "sudah", "aja": "saja", "trs": "terus", "bgt": "banget", "dlm": "dalam",
    "rp": "rupiah", "jt": "juta", "rb": "ribu", "cashback": "cashback", "promo": "promosi",
    "diskon": "diskon", "gratis": "gratis", "hadiah": "hadiah", "menang": "menang", "undian": "undian",
    "tlp": "telepon", "telp": "telepon", "hp": "handphone", "no": "nomor", "rek": "rekening",
    "an": "atas nama", "selamat!!!": "selamat", "selamat!!": "selamat", "selamat!": "selamat",
    "trnsfer": "transfer", "trf": "transfer", "4nda": "anda", "4pa": "apa", "1ni": "ini",
    "k4mu": "kamu", "klik": "klik", "segera": "segera", "buruan": "cepat", "cepat": "cepat",
    "terbatas": "terbatas", "khusus": "khusus", "free": "gratis", "win": "menang", "winner": "pemenang",
    "prize": "hadiah", "bonus": "bonus", "gift": "hadiah", "reward": "hadiah", "cash": "uang",
    "money": "uang", "credit": "kredit", "loan": "pinjaman", "bank": "bank", "transfer": "transfer",
    "account": "akun", "sspecial": "spesial", "balance": "saldo", "offer": "penawaran",
    "deal": "penawaran", "discount": "diskon", "sale": "diskon", "limited": "terbatas",
    "exclusive": "eksklusif", "click": "klik", "claim": "klaim", "buy": "beli", "order": "pesan",
    "register": "daftar", "subscribe": "langganan", "join": "gabung", "verify": "verifikasi",
    "confirm": "konfirmasi", "urgent": "segera", "now": "sekarang", "today": "hari ini",
    "instant": "instan", "software": "perangkat lunak", "system": "sistem", "update": "pembaruan",
    "download": "unduh", "install": "instal", "hello": "halo", "dear": "halo", "sir": "bapak",
    "madam": "ibu", "service": "layanan", "customer": "pelanggan", "support": "dukungan",
    "information": "informasi", "message": "pesan"
}

def normalize_text(text):
    words = text.split()
    normalized_words = [NORMALIZATION_DICT.get(word, word) for word in words]
    return ' '.join(normalized_words)

factory = StopWordRemoverFactory()
stopwords = set(factory.get_stop_words())

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stopwords])

def full_preprocess(text):
    return remove_stopwords(normalize_text(preprocess(str(text))))

df['clean'] = df['text'].map(full_preprocess)

print("Memuat Word2Vec model...")
w2v_model = Word2Vec.load('../program/w2v_kamus.model')

MAX_LEN = 50
EMBEDDING_DIM = 100

def get_sequence_vectors(text, max_len=MAX_LEN, embedding_dim=EMBEDDING_DIM):
    words = text.split()
    vector_seq = np.zeros((max_len, embedding_dim))
    
    for i, word in enumerate(words):
        if i >= max_len:
            break
        if word in w2v_model.wv.key_to_index:
            vector_seq[i] = w2v_model.wv[word]
            
    return vector_seq

print("Membuat urutan vektor (sequences)...")
X = np.array([get_sequence_vectors(t) for t in df['clean']])

le = LabelEncoder()
y = le.fit_transform(df['label']) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

print("Membangun model LSTM...")
model = Sequential([
    Input(shape=(MAX_LEN, EMBEDDING_DIM)),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

print("Melatih model LSTM...")
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

print("Mengevaluasi model pada test data...")
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred, target_names=le.classes_))

model_path = 'spam_model_lstm.keras'
model.save(model_path)
print(f"Model LSTM berhasil disimpan ke {model_path}!")

print(f"Label mapping: 0 -> {le.classes_[0]}, 1 -> {le.classes_[1]}")
