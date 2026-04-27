import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import torch
from gensim.models import Word2Vec
from pycaret.classification import load_model, predict_model

# Project root (parent of src/) for text_preprocess and deep_learning
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import text_preprocess  # noqa: E402
from text_preprocess import preprocess_stages  # noqa: E402
from deep_learning.model import (  # noqa: E402
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MAX_LEN,
    LSTMSpamModel,
)

# Artifacts: PyCaret + w2v in machine_learning/; LSTM in deep_learning/ (see deep_learning.train)
_ML_DIR = os.path.join(_ROOT, "machine_learning", "model")
_W2V_PATH = os.path.join(_ROOT, "machine_learning", "w2v_kamus.model")
_LSTM_PATH = os.path.join(_ROOT, "deep_learning", "model", "spam_model_lstm.pth")


# ===== LOAD MODELS (cached) =====
@st.cache_resource
def load_all_models():
    ml_models = {
        "SVM": load_model(os.path.join(_ML_DIR, "spam_model_svm")),
        "Naive Bayes": load_model(os.path.join(_ML_DIR, "spam_model_nb")),
        "Logistic Regression": load_model(os.path.join(_ML_DIR, "spam_model_lr")),
    }
    # Load PyTorch LSTM model
    device = torch.device("cpu")
    lstm_model = LSTMSpamModel()
    lstm_model.load_state_dict(torch.load(_LSTM_PATH, map_location=device, weights_only=True))
    lstm_model.eval()

    w2v = Word2Vec.load(_W2V_PATH)
    return ml_models, lstm_model, w2v


ml_models, lstm_model, w2v_model = load_all_models()

# Sequence shape must match LSTM training (deep_learning.model defaults)
MAX_LEN = DEFAULT_MAX_LEN
EMBEDDING_DIM = DEFAULT_EMBEDDING_DIM

feature_names = [f"w2v_{i}" for i in range(100)]


def document_vector(doc):
    valid_words = [word for word in doc if word in w2v_model.wv.key_to_index]
    if len(valid_words) == 0:
        return np.zeros(100)
    return np.mean(w2v_model.wv[valid_words], axis=0)


def sequence_vector(doc):
    seq = np.zeros((MAX_LEN, EMBEDDING_DIM))
    for i, word in enumerate(doc):
        if i >= MAX_LEN:
            break
        if word in w2v_model.wv.key_to_index:
            seq[i] = w2v_model.wv[word]
    return seq


def prediksi_email(teks, nama_model):
    """
    Preprocessing: ``text_preprocess.preprocess_stages`` (single source of truth with training data).
    """
    teks_bersih, teks_normal, teks_stopword = preprocess_stages(teks)
    kata_kata = teks_stopword.split()

    if nama_model == "LSTM":
        seq = torch.FloatTensor(sequence_vector(kata_kata)).unsqueeze(0)
        with torch.no_grad():
            prob = float(lstm_model(seq).item())
        label = "spam" if prob > 0.5 else "ham"
        return label, prob, teks_bersih, teks_normal, teks_stopword

    vec = document_vector(kata_kata)
    df_input = pd.DataFrame([vec], columns=feature_names)
    clf = ml_models[nama_model]
    hasil = predict_model(clf, data=df_input)
    label = hasil["prediction_label"].iloc[0]
    return str(label), None, teks_bersih, teks_normal, teks_stopword


# ===== TAMPILAN STREAMLIT =====
st.set_page_config(
    page_title="Deteksi Spam Email", page_icon="📧", layout="centered"
)

st.title("📧 Deteksi Spam Email")
st.markdown(
    "Masukkan teks email dan sistem akan memprediksi apakah email tersebut **SPAM** atau "
    "**HAM** (bukan spam)."
)

st.divider()

input_teks = st.text_area(
    "✉️ Masukkan teks email:",
    height=150,
    placeholder="Contoh: Selamat! Anda memenangkan hadiah 100 juta rupiah...",
)

pilihan_model = st.selectbox(
    "🤖 Pilih Model:",
    ["SVM", "Naive Bayes", "Logistic Regression", "LSTM"],
)

if st.button("🚀 Prediksi!", type="primary", use_container_width=True):
    if not input_teks.strip():
        st.warning("⚠️ Silakan masukkan teks email terlebih dahulu.")
    else:
        with st.spinner("Sedang menganalisis..."):
            label, prob, bersih, normal, stopword = prediksi_email(
                input_teks, pilihan_model
            )

        st.divider()

        is_spam = str(label).lower() == "spam"
        if is_spam:
            st.error("🚫 HASIL PREDIKSI: **SPAM**")
        else:
            st.success("✅ HASIL PREDIKSI: **HAM** (Bukan Spam)")

        if pilihan_model == "LSTM" and prob is not None:
            st.metric("Probabilitas spam (LSTM)", f"{prob:.1%}")

        with st.expander("🔍 Lihat detail proses preprocessing"):
            st.caption("Pipeline: `text_preprocess` (cleaning → normalisasi → stopword).")
            st.markdown(f"**Teks Asli:** {input_teks}")
            st.markdown(f"**Model Digunakan:** {pilihan_model}")

st.divider()
st.caption("Dibuat oleh Kelompok Kartini — PBA 2026")
