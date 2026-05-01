# Tugas: Pembersihan dataset untuk klasifikasi spam / ham

File data: `dataset/email_spam_indo.csv`.

## 1. Struktur data (wajib dipahami)

| Kolom     | Isi | Catatan |
|-----------|-----|---------|
| `Kategori` | Label kelas | Biasanya `spam` atau `ham` (periksa konsistensi huruf besar/kecil dan nilai lain). |
| `Pesan`   | Isi teks email | Satu sel = satu dokumen; bisa sangat panjang; sering mengandung header forward / balasan. |

**Yang harus dilakukan di level dataset (bukan hanya regex teks):**

- [ ] **Normalisasi label**: satukan bentuk (`spam` / `ham`) — hindari `Spam`, `HAM`, spasi ekstra. Drop atau perbaiki baris dengan label tidak dikenal.
- [ ] **Baris rusak / kosong**: hapus atau isi `Pesan` yang `NaN`, string kosong, atau hanya whitespace.
- [ ] **Duplikat**: pertimbangkan deduplikasi berdasarkan teks penuh atau hash (spam sering copy-paste).
- [ ] **Panjang teks**: putuskan batas minimum karakter/token; teks terlalu pendek bisa membingungkan model.
- [ ] **Imbang kelas**: cek rasio spam:ham; pertimbangkan stratified split atau teknik sampling jika tidak seimbang.

## 2. Polusi teks yang umum di file ini (dari sampel)

Berdasarkan isi `Pesan`, berikut noise yang relevan untuk klasifikasi:

| Jenis noise | Contoh / efek | Sudah ditangani oleh `text_preprocess.py`? |
|-------------|---------------|-------------------------------------------|
| URL / link | `http://...`, path web | Ya — dihapus di tahap `preprocess`. |
| Alamat email | `user@domain.com` | Ya — `\S+@\S+` dihapus. |
| Subjek / forward line (sebagian) | `Subjek: ...`, `... pada 09/20/2000 ...`, `Re: ...` | Sebagian — ada pola untuk subjek/forward tertentu; header panjang **ham** (Enron-style) masih banyak sisa teks repetitif. |
| Baris `Re:` / `Fw:` / `Fwd:` | Awal baris balasan | Ya — awal baris tipe `re|fw|fwd` dibersihkan (satu pola). |
| Token mirip domain | `something com` (bukan URL) | Ya — pola `\b\w+\s+(com|net|org|co|id)\b` (bisa agresif). |
| Angka, mata uang, tanda baca | `$ 454`, `100 %`, tanggal | Ya — setelah langkah lower, karakter non-huruf (ASCII) dibuang; **hanya huruf a–z dan spasi** tersisa. |
| Spasi / garis bawah berlebih | ` _ _ _ `, banyak spasi | Ya — runtuh jadi satu spasi. |
| Slang / typo / campuran EN-ID | `klik`, `free`, `CIalis`, dll. | Ditangani di **normalisasi kata** (`NORMALIZATION_DICT`), bukan di `preprocess` mentah. |
| Kata fungsi Indonesia | yang, dan, di, … | **Stopword** Sastrawi — dihapus di `full_preprocess` / pipeline training. |
| Email ham “forward” panjang | Header `diteruskan oleh ... Subjek:` + isi bisnis | Perlu evaluasi: pembersihan tambahan (lihat bagian 3) jika model fokus ke “isi” bukan metadata. |

## 3. Apa yang **perlu** Anda bersihkan / putuskan (tugas eksplisit)

### A. Koherensi dengan pipeline kode

- [ ] Pastikan **CSV** memakai nama kolom yang sama dengan yang dibaca loader: `Kategori` → `label`, `Pesan` → `text` (sudah di `deep_learning.dataloader.load_spam_email_dataframe`).
- [ ] Gunakan **satu pipeline** teks untuk training dan inference: `text_preprocess` → `full_preprocess` / `preprocess_stages` (jangan duplikasi aturan di notebook lain tanpa sinkronisasi).

### B. Perbaikan dataset (di luar kode saat ini)

- [ ] **Audit label**: pastikan tidak ada kelas salah (spam berlabel ham atau sebaliknya) pada sampel panjang/mirip.
- [ ] **Header email berulang** pada ham: pertimbangkan aturan tambahan (mis. potong segmen pertama `diteruskan oleh` / `forwarded by` dalam bahasa Indonesia & Inggris) — saat ini baru subbagian pola tertentu.
- [ ] **Teks spam “gibberish”** (kalimat tidak koheren): biasanya tetap spam; jangan dibuang hanya karena aneh, kecuali untuk kebisingan mutlak.
- [ ] **Privasi / PII** (opsional): jika akan dipublikasikan, redaksi nama/orang yang tidak perlu (bukan wajib untuk akurasi model).

### C. Evaluasi setelah pembersihan

- [ ] Simpan versi dataset bersih (mis. `email_spam_indo_clean.csv`) atau dokumentasikan commit hash + skrip transform.
- [ ] Ukur **baseline** (akurasi/F1) sebelum vs sesudah perubahan aturan, pada split yang sama (`random_state` tetap).

## 4. Ringkasan “harus dibersihkan” untuk klasifikasi spam/ham

| Prioritas | Apa | Mengapa |
|-----------|-----|---------|
| Tinggi | Label konsisten + baris valid | Tanpa ini model tidak belajar atau error. |
| Tinggi | URL, email, angka/tanda → pola huruf (sudah di kode) | Mengurangi fitur “ceroboh” dan domain spesifik. |
| Tinggi | Normalisasi kata + stopword (sudah di kode) | Menyamakan variasi spam Indonesia / Inggris. |
| Sedang | Forward/subject noise | Ham sering berisi header panjang; bisa mendominansi vektor. |
| Sedang | Duplikat & teks terlalu pendek | Data leakage / noise label. |
| Rendah | Stemming tambahan | Sastrawi tidak stem; tambahkan hanya jika eksperimen menunjukkan manfaat. |

## 5. File terkait di repo

- `text_preprocess.py` — aturan cleaning + normalisasi + stopword.
- `deep_learning/dataloader.py` — baca CSV, rename kolom, kolom `clean`.
- `machine_learning/train.py` — training SVM/NB/LR dari fitur yang sama.
- `deep_learning/train_lstm.py` — training LSTM dari teks `clean`.

Setelah tugas di atas selesai, jalankan ulang training dan bandingkan metrik di validasi/test set yang sama.
