import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ====================== PAGE CONFIG ======================
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
    background: rgba(255,255,255,0.80);
    padding: 2.5rem;
    border-radius: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}

/* WHITE INPUT FIELDS */
input, select, textarea {
    background-color: white !important;
    color: #1F2A44 !important;
}

.stSelectbox div, .stNumberInput input, .stTextInput input {
    background-color: white !important;
}

</style>
""", unsafe_allow_html=True)

# ====================== TITLE ======================
st.title("🌍💸 Digital Finance Access Predictor")
st.markdown("### East Africa • Kenya | Tanzania | Uganda")

# ====================== LOAD MODELS ======================
@st.cache_resource
def load_models():
    try:
        return {
            "pooled": joblib.load("model_pooled.joblib"),
            "experts": joblib.load("experts.joblib"),
            "gating": joblib.load("gating_model.joblib")
        }
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

models = load_models()

# ====================== GET TRUE FEATURE LIST ======================
FEATURES = list(models["pooled"].feature_names_in_)

# ====================== USER INPUT ======================
st.subheader("👤 User Profile")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 80, 32)
    female = st.radio("Gender", ["Male", "Female"])
    urban = st.radio("Location", ["Rural", "Urban"])

with col2:
    inc_q = st.selectbox("Income Quintile", [1,2,3,4,5])

    # ✅ EDUCATION FIX (READABLE LABELS)
    educ = st.selectbox(
        "Education Level",
        [0,1,2,3,4],
        format_func=lambda x: {
            0: "No Education",
            1: "Primary",
            2: "Secondary",
            3: "Tertiary",
            4: "Higher"
        }[x]
    )

    internet = st.radio("Internet Use", ["No","Yes"])

with col3:
    country = st.selectbox("Country", ["KEN","TZA","UGA"])

# ====================== BASE INPUT ======================
row = {
    "female": 1 if female == "Female" else 0,
    "age": age,
    "educ": educ,
    "inc_q": inc_q,
    "urbanicity": 1 if urban == "Urban" else 0,
    "internet_use": 1 if internet == "Yes" else 0,
    "dig_account": 0,
    "anydigpayment": 0,
    "wgt": 1.0
}

df = pd.DataFrame([row])

# ====================== ALIGN TO TRAINING SCHEMA ======================
for col in FEATURES:
    if col not in df.columns:
        df[col] = 0

df = df[FEATURES]

# ====================== MODEL PREDICTION ======================
def predict(df):

    pooled = models["pooled"].predict_proba(df)[0,1]

    country_pred = models["gating"].predict(df)[0]

    expert = pooled

    if country_pred in models["experts"]:
        try:
            expert = models["experts"][country_pred].predict_proba(df)[0,1]
        except:
            pass

    return pooled, expert

# ====================== RUN APP ======================
if st.button("🔮 Predict Digital Inclusion"):

    if models:

        try:
            pooled, expert = predict(df)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("🌐 Pooled Model", f"{pooled*100:.1f}%")

            with col2:
                st.metric("🧠 Expert Model", f"{expert*100:.1f}%")

            with col3:
                delta = expert - pooled
                st.metric("📊 Difference", f"{delta*100:+.1f}%")

            st.markdown("### 🧭 Interpretation")

            if abs(delta) < 0.05:
                st.info("Models are aligned.")
            elif delta > 0:
                st.success("Expert model is more optimistic.")
            else:
                st.warning("Expert model is more conservative.")

            final = expert

            if final > 0.75:
                st.success("🟢 High likelihood of digital inclusion")
            elif final > 0.5:
                st.warning("🟡 Moderate likelihood")
            else:
                st.error("🔴 Low likelihood")

        except Exception as e:
            st.error(f"Prediction error: {e}")

    else:
        st.error("Models not loaded")
