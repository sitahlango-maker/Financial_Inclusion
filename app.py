import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Digital Finance Access Predictor",
    page_icon="🌍💸",
    layout="wide"
)

# ====================== CLEAN SWIFT EXECUTIVE UI ======================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #F5F8FC 0%, #EEF3F9 50%, #E9EEF6 100%);
    color: #1F2A44;
}

.main .block-container {
    background: rgba(255,255,255,0.80);
    padding: 2.5rem;
    border-radius: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}

h1, h2, h3 {
    color: #1F4B7A;
}

/* WHITE INPUTS (CRITICAL FOR VISIBILITY) */
input, select, textarea {
    background-color: white !important;
    color: #1F2A44 !important;
}

/* STREAMLIT INPUT FIX */
.stTextInput input,
.stNumberInput input,
.stSelectbox div,
.stSlider {
    background-color: white !important;
    color: #1F2A44 !important;
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(90deg, #14B8A6, #0EA5A4);
    color: white;
    font-weight: 600;
    border-radius: 10px;
}

div[data-testid="metric-container"] {
    background: white;
    border-radius: 12px;
    padding: 12px;
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

# ====================== USER INPUT ======================
st.subheader("👤 User Profile")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 80, 32)
    female = st.radio("Gender", ["Male", "Female"])
    urban = st.radio("Location", ["Rural", "Urban"])

with col2:
    inc_q = st.selectbox("Income Quintile", [1, 2, 3, 4, 5])
    educ = st.selectbox("Education", [0,1,2,3,4])
    internet = st.radio("Internet Use", ["No","Yes"])

with col3:
    country = st.selectbox("Country", ["KEN","TZA","UGA"])

# ====================== MODEL FEATURES (EXACT MATCH) ======================
FEATURES = [
    "female","age","educ","inc_q","urbanicity",
    "dig_account","anydigpayment","internet_use","wgt",
    "reg_index","reg_cons_prot","reg_kyc_prop",
    "reg_entry_lim","reg_max_lim","reg_agent_el",
    "num_providers","earliest_launch"
]

# ====================== INPUT ENGINEERING ======================
row = {
    "female": 1 if female == "Female" else 0,
    "age": age,
    "educ": educ,
    "inc_q": inc_q,
    "urbanicity": 1 if urban == "Urban" else 0,
    "internet_use": 1 if internet == "Yes" else 0,
    "dig_account": 0,
    "anydigpayment": 0,
    "wgt": 1.0,

    # fixed country-level proxies (IMPORTANT: no country_code, no mmpi)
    "reg_index": 0.75 if country == "KEN" else 0.7,
    "reg_cons_prot": 0.7,
    "reg_kyc_prop": 1,
    "reg_entry_lim": 1,
    "reg_max_lim": 1,
    "reg_agent_el": 1,
    "num_providers": 4,
    "earliest_launch": 2012
}

df = pd.DataFrame([row])[FEATURES]

# ====================== MODEL LOGIC ======================
def predict_models(df):

    pooled = models["pooled"].predict_proba(df)[0, 1]

    country_pred = models["gating"].predict(df)[0]

    expert = pooled

    if country_pred in models["experts"]:
        try:
            expert = models["experts"][country_pred].predict_proba(df)[0, 1]
        except:
            pass

    return pooled, expert

# ====================== RUN PREDICTION ======================
if st.button("🔮 Predict Digital Inclusion"):

    if models:

        try:
            pooled, expert = predict_models(df)

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
                st.info("Models agree on prediction.")
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
