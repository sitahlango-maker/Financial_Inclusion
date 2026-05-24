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

# ====================== SWIFT-STYLE LIGHT UI ======================
st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background: linear-gradient(135deg, #F5F8FC 0%, #EEF3F9 50%, #E9EEF6 100%);
    color: #1F2A44;
}

/* MAIN CONTAINER */
.main .block-container {
    background: rgba(255, 255, 255, 0.78);
    border-radius: 18px;
    padding: 2.5rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    backdrop-filter: blur(10px);
}

/* HEADINGS */
h1 {
    color: #0F2D4A;
    font-weight: 700;
}
h2, h3 {
    color: #1F4B7A;
}

/* ================= WHITE INPUT BOXES (IMPORTANT FIX) ================= */
input, textarea, select {
    background-color: white !important;
    color: #1F2A44 !important;
    border-radius: 8px !important;
}

/* Streamlit widget overrides */
.stTextInput > div > div > input,
.stNumberInput input,
.stSelectbox div,
.stSlider,
.stRadio {
    background-color: white !important;
    color: #1F2A44 !important;
}

/* BUTTONS */
.stButton>button {
    background: linear-gradient(90deg, #14B8A6, #0EA5A4);
    color: white;
    border-radius: 10px;
    font-weight: 600;
}

/* METRICS */
div[data-testid="metric-container"] {
    background: white;
    border: 1px solid rgba(15,45,74,0.08);
    border-radius: 14px;
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
        models = {
            "pooled": joblib.load("model_pooled.joblib"),
            "experts": joblib.load("experts.joblib"),
            "gating": joblib.load("gating_model.joblib"),
            "feature_names": joblib.load("feature_names.joblib"),
            "medians": joblib.load("medians.joblib")
        }
        st.success("✅ Models loaded successfully")
        return models
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

models = load_models()

# ====================== INPUT FORM ======================
st.subheader("👤 User Profile")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 80, 32)
    female = st.radio("Gender", ["Male", "Female"], horizontal=True)
    urbanicity = st.radio("Location", ["Rural", "Urban"], horizontal=True)

with col2:
    inc_q = st.selectbox("Income Quintile", [1, 2, 3, 4, 5])
    educ = st.selectbox(
        "Education Level",
        [0, 1, 2, 3, 4],
        format_func=lambda x: ["None", "Primary", "Secondary", "Tertiary", "Higher"][x]
    )
    internet_use = st.radio("Internet Use", ["No", "Yes"], horizontal=True)

with col3:
    country_input = st.selectbox("Country", ["KEN", "TZA", "UGA"])
    country_code = country_input

# ====================== INPUT ENGINEERING ======================
input_dict = {
    "female": 1 if female == "Female" else 0,
    "age": age,
    "educ": educ,
    "inc_q": inc_q,
    "urbanicity": 1 if urbanicity == "Urban" else 0,
    "internet_use": 1 if internet_use == "Yes" else 0,
    "dig_account": 0,
    "anydigpayment": 0,
    "wgt": 1.0
}

country_features = {
    "KEN": {"reg_index": 0.85, "reg_cons_prot": 0.70, "reg_kyc_prop": 1,
            "reg_entry_lim": 1, "reg_max_lim": 1, "reg_agent_el": 1,
            "num_providers": 5, "earliest_launch": 2010},

    "TZA": {"reg_index": 0.72, "reg_cons_prot": 0.65, "reg_kyc_prop": 1,
            "reg_entry_lim": 1, "reg_max_lim": 0, "reg_agent_el": 1,
            "num_providers": 4, "earliest_launch": 2014},

    "UGA": {"reg_index": 0.78, "reg_cons_prot": 0.68, "reg_kyc_prop": 1,
            "reg_entry_lim": 1, "reg_max_lim": 1, "reg_agent_el": 0,
            "num_providers": 4, "earliest_launch": 2012}
}

input_dict.update(country_features.get(country_code, {}))

input_df = pd.DataFrame([input_dict])

# ====================== ALIGN FEATURES (CRITICAL FIX) ======================
def align_features(df):
    for col in models["feature_names"]:
        if col not in df.columns:
            df[col] = 0
    return df[models["feature_names"]]

# ====================== MODEL COMPARISON ======================
def predict_models(df):

    df = align_features(df)

    pooled_prob = models["pooled"].predict_proba(df)[0, 1]

    pred_country = models["gating"].predict(df)[0]

    expert_prob = pooled_prob

    if pred_country in models["experts"]:
        try:
            expert = models["experts"][pred_country]

            sample = df.copy()

            if hasattr(expert, "feature_names_in_"):
                sample = sample.reindex(columns=expert.feature_names_in_, fill_value=0)

            expert_prob = expert.predict_proba(sample)[0, 1]

        except:
            expert_prob = pooled_prob

    return pooled_prob, expert_prob

# ====================== PREDICTION ======================
if st.button("🔮 Predict Digital Inclusion", type="primary"):

    if models:

        try:
            pooled_prob, expert_prob = predict_models(input_df)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("🌐 Pooled Model", f"{pooled_prob*100:.1f}%")

            with col2:
                st.metric("🧠 Expert Model", f"{expert_prob*100:.1f}%")

            with col3:
                delta = expert_prob - pooled_prob
                st.metric("📊 Difference", f"{delta*100:+.1f}%")

            st.markdown("### 🧭 Interpretation")

            if abs(delta) < 0.05:
                st.info("Models are aligned.")
            elif delta > 0:
                st.success("Expert model is more optimistic.")
            else:
                st.warning("Expert model is more conservative.")

            final_prob = expert_prob

            if final_prob > 0.75:
                st.success("🟢 High likelihood of inclusion")
            elif final_prob > 0.5:
                st.warning("🟡 Moderate likelihood")
            else:
                st.error("🔴 Low likelihood")

        except Exception as e:
            st.error(f"Prediction error: {e}")

    else:
        st.error("Models not loaded")
