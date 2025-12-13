import streamlit as st
import joblib
import re

# ==============================
# Load Model & Vectorizer
# ==============================

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/review_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_artifacts()

# ==============================
# Text Cleaning Function
# ==============================

def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

# ==============================
# UI
# ==============================

st.set_page_config(page_title="ReviewGuard", page_icon="🔍")

st.title("🔍 ReviewGuard: Fake Review Detector")
st.write("Enter a review below to check whether it is **fake or genuine**.")

review_text = st.text_area("✍️ Review Text", height=150)

# ==============================
# Prediction
# ==============================

if st.button("Analyze Review"):
    if review_text.strip() == "":
        st.warning("Please enter some review text.")
    else:
        cleaned = clean_text(review_text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized).max()

        if prediction == 1:
            st.error(f"🔴 **FAKE REVIEW** (Confidence: {confidence:.1%})")
            st.warning("⚠️ This review shows deceptive language patterns.")
            st.write("**Why?**")
            st.write("- Overly generic or exaggerated language")
            st.write("- Lack of concrete experience details")
            st.write("- Promotional tone")
        else:
            st.success(f"🟢 **GENUINE REVIEW** (Confidence: {confidence:.1%})")
            st.info("✓ This review appears authentic.")
            st.write("**Why?**")
            st.write("- Balanced opinion")
            st.write("- Mentions real usage details")
            st.write("- Natural wording")

# ==============================
# Sidebar Examples
# ==============================

st.sidebar.header("🧪 Test Examples")

if st.sidebar.button("Example: Fake Review"):
    st.session_state.example = (
        "This is the BEST product ever!!! Absolutely amazing!!! "
        "Everyone must buy this right now!!! Five stars!!!"
    )

if st.sidebar.button("Example: Genuine Review"):
    st.session_state.example = (
        "I stayed here for two nights during a business trip. "
        "The room was clean and quiet, though the WiFi was a bit slow. "
        "Location was convenient and staff were helpful."
    )

if "example" in st.session_state:
    st.text_area("✍️ Review Text", st.session_state.example, height=150)
