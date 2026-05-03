import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Digital Financial Inclusion Predictor",
    layout="wide"
)

st.title("🌍 Digital Financial Inclusion Predictor")
st.markdown("Analyze digital financial access using country-specific or pooled models.")

# ===============================
# LOAD MODELS
# ===============================
BASE_DIR = os.path.dirname(__file__)

@st.cache_resource
def load_models():
    model_pooled = joblib.load(os.path.join(BASE_DIR, "model_pooled.pkl"))
    experts = joblib.load(os.path.join(BASE_DIR, "experts.pkl"))
    feature_names = joblib.load(os.path.join(BASE_DIR, "feature_names.pkl"))
    return model_pooled, experts, feature_names

model_pooled, experts, feature_names = load_models()

# ===============================
# SESSION STATE
# ===============================
if "history" not in st.session_state:
    st.session_state.history = []

# ===============================
# INPUT UI
# ===============================
st.subheader("👤 Enter User Profile")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 80, 30)
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    urbanicity = st.radio("Location", ["Rural", "Urban"], horizontal=True)

with col2:
    inc_q = st.selectbox("Income Quintile", [1, 2, 3, 4, 5])
    educ = st.selectbox(
        "Education Level",
        options=[0, 1, 2, 3, 4],
        format_func=lambda x: ["No Education", "Primary", "Secondary", "Tertiary", "Higher"][x]
    )
    internet_use = st.radio("Uses Internet", ["No", "Yes"], horizontal=True)

with col3:
    model_choice = st.selectbox(
        "Select Model",
        ["POOLED", "KEN", "TZA", "UGA"]
    )

# ===============================
# PREP INPUT
# ===============================
input_dict = {
    "age": age,
    "female": 1 if gender == "Female" else 0,
    "urbanicity": 1 if urbanicity == "Urban" else 0,
    "inc_q": inc_q,
    "educ": educ,
    "internet_use": 1 if internet_use == "Yes" else 0,
    "country_code": model_choice if model_choice != "POOLED" else "KEN"
}

input_df = pd.DataFrame([input_dict])

# Align features
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[feature_names]

# ===============================
# RUN MODEL
# ===============================
if st.button("🔮 Run Analysis", type="primary"):

    try:
        # Select model
        if model_choice == "POOLED":
            model = model_pooled
            model_name = "Pooled Model"
        else:
            model = experts[model_choice]
            model_name = f"Expert Model ({model_choice})"

        prob = model.predict_proba(input_df)[0, 1]

        # Save to history
        st.session_state.history.append({
            "Model": model_name,
            "Probability": prob
        })

        # ===============================
        # COLOR LOGIC
        # ===============================
        if prob >= 0.75:
            color = "green"
            label = "High likelihood of access"
        elif prob >= 0.50:
            color = "orange"
            label = "Moderate likelihood of access"
        else:
            color = "red"
            label = "Low likelihood of access"

        # ===============================
        # RESULT DISPLAY
        # ===============================
        st.markdown("## 🎯 Prediction Result")

        st.markdown(f"""
        <div style="
            padding:25px;
            border-radius:12px;
            background:#f9fafb;
            border:2px solid {color};
            text-align:center;
        ">
            <h3 style="color:#111;">Probability of Access</h3>
            <h1 style="color:{color}; font-size:52px; margin:10px 0;">
                {prob:.1%}
            </h1>
            <p style="font-size:18px; color:{color};">
                {label}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.metric("Model Used", model_name)

    except Exception as e:
        st.error(f"Error: {e}")

# ===============================
# COMPARISON PANEL
# ===============================
st.markdown("## 📊 Comparison Panel")

if len(st.session_state.history) > 0:

    df_hist = pd.DataFrame(st.session_state.history)

    st.dataframe(df_hist, use_container_width=True)

    for i, row in df_hist.iterrows():
        st.progress(row["Probability"])
        st.write(f"{row['Model']} → {row['Probability']:.1%}")

    if st.button("🧹 Clear Comparison"):
        st.session_state.history = []

else:
    st.info("Run models to compare results here.")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.markdown(
    "<center>Digital Financial Inclusion Predictor • Model Explorer</center>",
    unsafe_allow_html=True
)
