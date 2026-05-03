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
st.markdown("Mixture of Experts model for predicting digital financial access in East Africa.")

# ===============================
# LOAD MODELS
# ===============================
BASE_DIR = os.path.dirname(__file__)

@st.cache_resource
def load_models():
    model_pooled = joblib.load(os.path.join(BASE_DIR, "model_pooled.pkl"))
    gating_model = joblib.load(os.path.join(BASE_DIR, "gating_model.pkl"))
    experts = joblib.load(os.path.join(BASE_DIR, "experts.pkl"))
    feature_names = joblib.load(os.path.join(BASE_DIR, "feature_names.pkl"))
    return model_pooled, gating_model, experts, feature_names

model_pooled, gating_model, experts, feature_names = load_models()

# ===============================
# INPUT SECTION
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
        options=[0, 1, 2, 3, 4],
        format_func=lambda x: ["No Education", "Primary", "Secondary", "Tertiary", "Higher"][x]
    )
    internet_use = st.radio("Uses Internet", ["No", "Yes"], horizontal=True)

with col3:
    country = st.selectbox("Country", ["KEN (Kenya)", "TZA (Tanzania)", "UGA (Uganda)"])
    country_code = country.split()[0]

# ===============================
# PREPARE INPUT
# ===============================
input_dict = {
    "age": age,
    "female": 1 if gender == "Female" else 0,
    "urbanicity": 1 if urbanicity == "Urban" else 0,
    "inc_q": inc_q,
    "educ": educ,
    "internet_use": 1 if internet_use == "Yes" else 0,
    "country_code": country_code
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

def box(title, value, color_code):
    st.markdown(f"""
        <div style="
            background-color:#0f172a;
            padding:20px;
            border-radius:12px;
            text-align:center;
            margin-bottom:10px;">
            <h4 style="color:#ffffff;">{title}</h4>
            <h1 style="color:{color_code};">{value:.1%}</h1>
        </div>
    """, unsafe_allow_html=True)

# ===============================
# PREDICTION
# ===============================
if st.button("🔮 Predict Digital Financial Access", type="primary"):

    with st.spinner("Running model..."):

        try:
            # Gating model
            pred_country = gating_model.predict(input_df)[0]
            gating_conf = np.max(gating_model.predict_proba(input_df))

            # Expert model
            expert_prob = None
            expert_label = "No Expert Used"

            if pred_country in experts and gating_conf > 0.4:
                expert_prob = experts[pred_country].predict_proba(input_df)[0, 1]
                expert_label = f"Expert ({pred_country})"

            # Pooled model (ALWAYS USED FOR COMPARISON)
            pooled_prob = model_pooled.predict_proba(input_df)[0, 1]

            # Final decision
            if expert_prob is not None:
                final_prob = expert_prob
                model_name = expert_label
            else:
                final_prob = pooled_prob
                model_name = "Pooled Model"

            # ===============================
            # MAIN RESULT
            # ===============================
            st.markdown("## 🎯 Prediction Result")

            colA, colB = st.columns(2)

            with colA:
                box("Probability of Access", final_prob, color(final_prob))

            with colB:
                st.metric("Model Used", model_name)
                st.metric("Gating Confidence", f"{gating_conf:.1%}")

            # Interpretation
            if final_prob >= 0.75:
                st.success("🟢 High likelihood of access")
            elif final_prob >= 0.50:
                st.info("🟠 Moderate likelihood of access")
            else:
                st.error("🔴 Low likelihood — barriers exist")

            # ===============================
            # COMPARISON (POOLED ALWAYS SHOWN)
            # ===============================
            st.markdown("## 📊 Model Comparison")

            c1, c2 = st.columns(2)

            with c1:
                box("Pooled Model", pooled_prob, color(pooled_prob))

            with c2:
                if expert_prob is not None:
                    box(f"Expert Model ({pred_country})", expert_prob, color(expert_prob))
                else:
                    st.warning("No expert model used for this prediction")

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.markdown(
    "<center>Digital Financial Inclusion Predictor • Mixture of Experts</center>",
    unsafe_allow_html=True
)
