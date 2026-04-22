import streamlit as st
import numpy as np
import re
import os
import tensorflow as tf
from gensim.models import Word2Vec
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ===== PATH (KHUSUS COLAB) =====
BASE_DIR = "/content"

# ===== LOAD MODEL =====
@st.cache_resource
def load_all():
    model = tf.keras.models.load_model(os.path.join(BASE_DIR, "spam_model_lstm.keras"))
    w2v = Word2Vec.load(os.path.join(BASE_DIR, "w2v_kamus.model"))
    return model, w2v

model, w2v_model = load_all()

# ===== PARAMETER =====
MAX_LEN = 50
EMBEDDING_DIM = 100

# ===== PREPROCESSING =====
def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_text(text):
    return text

factory = StopWordRemoverFactory()
stopwords = set(factory.get_stop_words())

def remove_stopwords(text):
    return ' '.join([w for w in text.split() if w not in stopwords])

# ===== SEQUENCE =====
def get_sequence_vectors(text):
    words = text.split()
    seq = np.zeros((MAX_LEN, EMBEDDING_DIM))

    for i, word in enumerate(words):
        if i >= MAX_LEN:
            break
        if word in w2v_model.wv.key_to_index:
            seq[i] = w2v_model.wv[word]

    return seq

# ===== PREDIKSI =====
def predict_email(teks):
    bersih = preprocess(teks)
    normal = normalize_text(bersih)
    stopword = remove_stopwords(normal)

    seq = get_sequence_vectors(stopword)
    seq = np.expand_dims(seq, axis=0)

    prob = model.predict(seq)[0][0]
    label = "SPAM" if prob > 0.5 else "HAM"

    return label, prob

# ===== UI =====
st.title("📧 Deteksi Spam Email (LSTM)")

input_teks = st.text_area("Masukkan teks email:")

if st.button("Prediksi"):
    if input_teks.strip() == "":
        st.warning("Masukkan teks dulu!")
    else:
        label, prob = predict_email(input_teks)

        if label == "SPAM":
            st.error(f"SPAM ({prob:.2%})")
        else:
            st.success(f"HAM ({prob:.2%})")

        st.progress(float(prob))
