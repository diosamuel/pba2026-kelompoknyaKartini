---
title: Spam Indonesia Email Detect
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: spam-indonesia-email-detect
license: mit
---

<div align="center">

# Deteksi Spam Email Indonesia

### Pemrosesan Bahasa Alami — Kelompok Kartini

Klasifikasi email **Spam** atau **Ham** (bukan spam) berbahasa Indonesia menggunakan NLP dan Machine Learning.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![HuggingFace Space](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Space-blue)](https://huggingface.co/spaces/diosamuel/spam-indonesia-email-detect)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Fitur

- Preprocessing teks lengkap (cleaning, normalisasi slang, stopword removal)
- Representasi teks menggunakan **Word2Vec**
- Tiga model klasifikasi: **SVM**, **Naive Bayes**, **Logistic Regression**
- Antarmuka web interaktif via Streamlit
- Deploy-ready dengan Docker & HuggingFace Spaces

## Tech Stack

| Komponen | Teknologi |
|---|---|
| Framework | Streamlit |
| NLP | Sastrawi, Gensim (Word2Vec) |
| ML | PyCaret, scikit-learn |
| Deploy | Docker, HuggingFace Spaces |

## Cara Menjalankan

```bash
# Clone repository
git clone https://github.com/diosamuel/pba2026-kelompoknyaKartini.git
cd pba2026-kelompoknyaKartini

# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run src/program/app.py
```

Atau dengan Docker:

```bash
docker build -t spam-email-detect .
docker run -p 8501:8501 spam-email-detect
```

## Anggota Kelompok

| Nama | NIM | GitHub |
|---|---|---|
| Virdio Samuel Saragih | 122450124 | [@diosamuel](https://github.com/diosamuel) |
| Baruna Abirawa | 122450097 | [@barunaxyz](https://github.com/barunaxyz) |
| Kartini Lovian Simbolon | 122450003 | [@kartinils](https://github.com/kartinils) |

## Links

| | URL |
|---|---|
| GitHub | [pba2026-kelompoknyaKartini](https://github.com/diosamuel/pba2026-kelompoknyaKartini) |
| HuggingFace Model | [spam-indonesia-email-detect](https://huggingface.co/diosamuel/spam-indonesia-email-detect) |
| HuggingFace Space | [Live Demo](https://huggingface.co/spaces/diosamuel/spam-indonesia-email-detect) |

---

<div align="center">

Dibuat oleh **Kelompok Kartini** — PBA 2026

</div>
