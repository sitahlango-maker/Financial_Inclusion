import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ====================== CONFIG ======================
st.set_page_config(
    page_title="Digital Finance Access Predictor",
    page_icon="🌍💸",
    layout="wide"
)

# ====================== CLEAN UI ======================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #F5F8FC 0%, #EEF3F9 50%, #E9EEF6 100%);
}

.main .block-container {
    background: rgba(255,255,255,0.78);
    padding: 2.5rem;
    border-radius: 18px;
}

input, select {
    background: white !important;
    color: black !important;
}

</style>
""", unsafe_allow_html=True)

st.title("🌍💸 Digital Finance Access Predictor")
st.markdown("### East Africa • Kenya | Tanzania | Uganda")

# ====================== LOAD MODELS ======================
@st.cache_resource
def load_models():
    return {
        "pooled": joblib.load("model_pooled.joblib"),
        "experts": joblib.load("experts.joblib"),
        "gating": joblib.load("gating_model.joblib"),
        "medians": joblib.load("medians.joblib")
    }

models = load_models()

# ====================== INPUT ======================
st.subheader("👤 User Profile")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 80, 32)
    female = st.radio("Gender", ["Male", "Female"])
    urban = st.radio("Location", ["Rural", "Urban"])

with col2:
    inc_q = st.selectbox("Income Quintile", [1,2,3,4,5])
    educ = st.selectbox("Education", [0,1,2,3,4])
    internet = st.radio("Internet Use", ["No","Yes"])

with col3:
    country = st.selectbox("Country", ["KEN","TZA","UGA"])

# ====================== BASE INPUT ======================
row = {
    "female": 1 if female=="Female" else 0,
    "age": age,
    "educ": educ,
    "inc_q": inc_q,
    "urbanicity": 1 if urban=="Urban" else 0,
    "internet_use": 1 if internet=="Yes" else 0,
    "dig_account": 0,
    "anydigpayment": 0,
    "wgt": 1.0,
    "country_code": country,
    "mmpi_2023": 0.7 if country=="KEN" else 0.6
}

df = pd.DataFrame([row])

# ====================== ALIGN TO TRAINING SCHEMA ======================
FEATURES = models["pooled"].feature_names_in_

def align(df):
    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0
    return df[FEATURES]

# ====================== MODEL FUNCTION ======================
def predict(df):

    df = align(df)

    pooled = models["pooled"].predict_proba(df)[0,1]

    country = models["gating"].predict(df)[0]

    expert = pooled

    if country in models["experts"]:
        try:
            m = models["experts"][country]
            expert = m.predict_proba(df)[0,1]
        except:
            pass

    return pooled, expert

# ====================== RUN ======================
if st.button("🔮 Predict"):

    try:
        pooled, expert = predict(df)

        st.metric("Pooled Model", f"{pooled*100:.1f}%")
        st.metric("Expert Model", f"{expert*100:.1f}%")

        diff = expert - pooled

        if abs(diff) < 0.05:
            st.info("Models agree")
        elif diff > 0:
            st.success("Expert more optimistic")
        else:
            st.warning("Expert more conservative")

    except Exception as e:
        st.error(f"Prediction error: {e}")
