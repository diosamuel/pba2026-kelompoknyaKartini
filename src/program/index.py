import streamlit as st

from tool import predict_spam, train_model


@st.cache_resource
def load_model():
    return train_model()


st.set_page_config(page_title="Deteksi Spam Email", page_icon="📧", layout="centered")

st.title("📧 Deteksi Spam Email")
st.caption("Masukkan isi email di bawah, lalu klik tombol untuk mendeteksi apakah email tersebut spam atau bukan.")

w2v_model, svm_model = load_model()

email_text = st.text_area(
    "Isi Email",
    height=200,
    placeholder="Tempel atau ketik isi email di sini...",
)

detect_clicked = st.button("🔍 Deteksi", type="primary", use_container_width=True)

if detect_clicked:
    if not email_text.strip():
        st.warning("Silakan masukkan teks email terlebih dahulu.")
    else:
        with st.spinner("Menganalisis email..."):
            label, confidence = predict_spam(email_text, w2v_model, svm_model)

        if label == "SPAM":
            st.error(f"**Hasil: {label}**")
        else:
            st.success(f"**Hasil: {label}**")

        st.metric("Confidence", f"{confidence:.1%}")
