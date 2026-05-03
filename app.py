import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
import tempfile

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Digital Financial Inclusion Predictor",
    layout="wide"
)

st.title("🌍 Digital Financial Inclusion Predictor")
st.markdown("Mixture of Experts + Pooled Model Comparison")

# ===============================
# LOAD MODELS FROM GITHUB
# ===============================
BASE_URL = "https://raw.githubusercontent.com/sitahlango-maker/Financial_Inclusion/main/"

def load_model(file_name):
    url = BASE_URL + file_name
    response = requests.get(url)

    if response.status_code != 200:
        st.error(f"Failed to load {file_name}")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    return joblib.load(tmp_path)

@st.cache_resource
def load_all():
    model_pooled = load_model("model_pooled.pkl")
    gating_model = load_model("gating_model.pkl")
    experts = load_model("experts.pkl")
    feature_names = load_model("feature_names.pkl")
    return model_pooled, gating_model, experts, feature_names

model_pooled, gating_model, experts, feature_names = load_all()

st.success("Models loaded successfully ✅")

# ===============================
# INPUT UI
# ===============================
st.subheader("👤 Enter User Profile")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 80, 32)
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    urbanicity = st.radio("Location", ["Rural", "Urban"], horizontal=True)

with col2:
    inc_q = st.selectbox("Income Quintile", [1, 2, 3, 4, 5])
    educ = st.selectbox(
        "Education Level",
        [0, 1, 2, 3, 4],
        format_func=lambda x: ["No Education", "Primary", "Secondary", "Tertiary", "Higher"][x]
    )
    internet_use = st.radio("Uses Internet", ["No", "Yes"], horizontal=True)

with col3:
    country = st.selectbox("Country", ["KEN", "TZA", "UGA"])

# ===============================
# INPUT PREPARATION
# ===============================
input_dict = {
    "age": age,
    "female": 1 if gender == "Female" else 0,
    "urbanicity": 1 if urbanicity == "Urban" else 0,
    "inc_q": inc_q,
    "educ": educ,
    "internet_use": 1 if internet_use == "Yes" else 0
}

input_df = pd.DataFrame([input_dict])

for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[feature_names]

# ===============================
# COLOR FUNCTION
# ===============================
def color(p):
    if p >= 0.75:
        return "#16a34a"
    elif p >= 0.5:
        return "#f59e0b"
    else:
        return "#dc2626"

def box(title, value, c):
    st.markdown(f"""
    <div style="
        background-color:#ffffff;
        padding:18px;
        border-radius:12px;
        border:1px solid #ddd;
        text-align:center;">
        <h4>{title}</h4>
        <h1 style="color:{c};">{value:.1%}</h1>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# PREDICTION
# ===============================
if st.button("🔮 Predict", type="primary"):

    try:
        # ===========================
        # GATING MODEL (country route)
        # ===========================
        pred_country = gating_model.predict(input_df)[0]
        gating_conf = np.max(gating_model.predict_proba(input_df))

        # ===========================
        # EXPERT MODEL
        # ===========================
        if pred_country in experts and gating_conf > 0.4:
            expert_prob = experts[pred_country].predict_proba(input_df)[0, 1]
            expert_model_name = f"Expert ({pred_country})"
        else:
            expert_prob = model_pooled.predict_proba(input_df)[0, 1]
            expert_model_name = "Pooled (Fallback)"

        # ===========================
        # POOLED MODEL (ALWAYS RUN)
        # ===========================
        pooled_prob = model_pooled.predict_proba(input_df)[0, 1]

        # ===========================
        # RESULTS
        # ===========================
        st.markdown("## 🎯 Results Comparison")

        colA, colB, colC = st.columns(3)

        with colA:
            box("Expert Model", expert_prob, color(expert_prob))
            st.caption(expert_model_name)

        with colB:
            box("Pooled Model", pooled_prob, color(pooled_prob))
            st.caption("Global Model")

        with colC:
            st.metric("Gating Confidence", f"{gating_conf:.1%}")
            st.metric("Predicted Country", pred_country)

        # ===========================
        # INTERPRETATION (based on expert)
        # ===========================
        if expert_prob >= 0.75:
            st.success("🟢 High likelihood of access")
        elif expert_prob >= 0.5:
            st.info("🟠 Moderate likelihood")
        else:
            st.error("🔴 Low likelihood")

    except Exception as e:
        st.error(f"Error: {e}")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.markdown("Digital Financial Inclusion Predictor • Pooled vs Expert Comparison")
