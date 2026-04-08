import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from gensim.models import Word2Vec
from pycaret.classification import load_model, predict_model
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ===== PATH KE MODEL =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== LOAD MODEL (sekali saja saat pertama kali) =====
@st.cache_resource
def load_all_models():
    models = {
        "SVM": load_model(os.path.join(BASE_DIR, 'spam_model_svm')),
        "Naive Bayes": load_model(os.path.join(BASE_DIR, 'spam_model_nb')),
        "Logistic Regression": load_model(os.path.join(BASE_DIR, 'spam_model_lr')),
    }
    w2v = Word2Vec.load(os.path.join(BASE_DIR, 'w2v_kamus.model'))
    return models, w2v

models, w2v_model = load_all_models()

# ===== PREPROCESSING (100% dari preprocessing_nlp.ipynb) =====

# Cell 2: preprocess
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

# Cell 3: NORMALIZATION_DICT + normalize_text
NORMALIZATION_DICT = {
    # Singkatan umum
    "gk": "tidak",
    "d": "di",
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

    # Kata khas spam / promo
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

    # Variasi kata penting
    "tlp": "telepon",
    "telp": "telepon",
    "hp": "handphone",
    "no": "nomor",
    "rek": "rekening",
    "an": "atas nama",

    # Variasi penulisan spam
    "selamat!!!": "selamat",
    "selamat!!": "selamat",
    "selamat!": "selamat",
    "trnsfer": "transfer",
    "trf": "transfer",

    # Variasi angka ke huruf (leetspeak)
    "4nda": "anda",
    "4pa": "apa",
    "1ni": "ini",
    "k4mu": "kamu",

    # Kata clickbait spam
    "klik": "klik",
    "segera": "segera",
    "buruan": "cepat",
    "cepat": "cepat",
    "terbatas": "terbatas",
    "khusus": "khusus",

    # Kata inti spam
    "free": "gratis",
    "win": "menang",
    "winner": "pemenang",
    "prize": "hadiah",
    "bonus": "bonus",
    "gift": "hadiah",
    "reward": "hadiah",

    # Finansial / uang
    "cash": "uang",
    "money": "uang",
    "credit": "kredit",
    "loan": "pinjaman",
    "bank": "bank",
    "transfer": "transfer",
    "account": "akun",
    "sspecial": "spesial",
    "balance": "saldo",

    # Promosi / marketing
    "offer": "penawaran",
    "deal": "penawaran",
    "discount": "diskon",
    "sale": "diskon",
    "limited": "terbatas",
    "exclusive": "eksklusif",

    # Call to action
    "click": "klik",
    "claim": "klaim",
    "buy": "beli",
    "order": "pesan",
    "register": "daftar",
    "subscribe": "langganan",
    "join": "gabung",
    "verify": "verifikasi",
    "confirm": "konfirmasi",

    # urgency
    "urgent": "segera",
    "now": "sekarang",
    "today": "hari ini",
    "instant": "instan",

    # teknologi / produk
    "software": "perangkat lunak",
    "system": "sistem",
    "update": "pembaruan",
    "download": "unduh",
    "install": "instal",

    # email umum
    "hello": "halo",
    "dear": "halo",
    "sir": "bapak",
    "madam": "ibu",

    # lain-lain yang sering muncul
    "service": "layanan",
    "customer": "pelanggan",
    "support": "dukungan",
    "information": "informasi",
    "message": "pesan"
}

def normalize_text(text):
    words = text.split()
    normalized_words = [NORMALIZATION_DICT.get(word, word) for word in words]
    return ' '.join(normalized_words)

# Cell 5: remove_stopwords (Sastrawi murni)
factory = StopWordRemoverFactory()
stopwords = set(factory.get_stop_words())

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stopwords])

# Cell 8: document_vector (Word2Vec)
def document_vector(doc):
    valid_words = [word for word in doc if word in w2v_model.wv.key_to_index]
    if len(valid_words) == 0:
        return np.zeros(100)
    return np.mean(w2v_model.wv[valid_words], axis=0)

feature_names = [f"w2v_{i}" for i in range(100)]

# ===== FUNGSI PREDIKSI UTAMA =====
def prediksi_email(teks, nama_model):
    teks_bersih = preprocess(teks)
    teks_normal = normalize_text(teks_bersih)
    teks_stopword = remove_stopwords(teks_normal)
    kata_kata = teks_stopword.split()
    vektor = document_vector(kata_kata)
    df_input = pd.DataFrame([vektor], columns=feature_names)
    
    model = models[nama_model]
    hasil = predict_model(model, data=df_input)
    label = hasil['prediction_label'].iloc[0]
    return label, teks_bersih, teks_normal, teks_stopword

# ===== TAMPILAN STREAMLIT =====
st.set_page_config(page_title="Deteksi Spam Email", page_icon="📧", layout="centered")

st.title("📧 Deteksi Spam Email")
st.markdown("Masukkan teks email dan sistem akan memprediksi apakah email tersebut **SPAM** atau **HAM** (bukan spam).")

st.divider()

# Input teks
input_teks = st.text_area("✉️ Masukkan teks email:", height=150,
    placeholder="Contoh: Selamat! Anda memenangkan hadiah 100 juta rupiah...")

# Pilih model
pilihan_model = st.selectbox("🤖 Pilih Model:", ["SVM", "Naive Bayes", "Logistic Regression"])

# Tombol prediksi
if st.button("🚀 Prediksi!", type="primary", use_container_width=True):
    if input_teks.strip() == "":
        st.warning("⚠️ Silakan masukkan teks email terlebih dahulu.")
    else:
        with st.spinner("Sedang menganalisis..."):
            label, bersih, normal, stopword = prediksi_email(input_teks, pilihan_model)
        
        st.divider()
        
        if label.lower() == "spam":
            st.error(f"🚫 HASIL PREDIKSI: **SPAM**")
        else:
            st.success(f"✅ HASIL PREDIKSI: **HAM** (Bukan Spam)")
        
        # Detail proses
        with st.expander("🔍 Lihat detail proses preprocessing"):
            st.markdown(f"**Teks Asli:** {input_teks}")
            st.markdown(f"**Setelah Cleaning:** {bersih}")
            st.markdown(f"**Setelah Normalisasi:** {normal}")
            st.markdown(f"**Setelah Stopword Removal:** {stopword}")
            st.markdown(f"**Model Digunakan:** {pilihan_model}")

st.divider()
st.caption("Dibuat oleh Kelompok Kartini — PBA 2026")
