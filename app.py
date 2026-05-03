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
# COLOR FUNCTION (IMPROVED CONTRAST)
# ===============================
def get_color(p):
    if p >= 0.75:
        return "#22c55e"  # green
    elif p >= 0.5:
        return "#f59e0b"  # amber
    else:
        return "#ef4444"  # red

# ===============================
# PREDICTION
# ===============================
if st.button("🔮 Predict Digital Financial Access", type="primary"):

    with st.spinner("Running model..."):

        try:
            # Gating model
            pred_country = gating_model.predict(input_df)[0]
            gating_conf = np.max(gating_model.predict_proba(input_df))

            # Expert selection
            expert_prob = None
            expert_model_used = False

            if pred_country in experts and gating_conf > 0.4:
                expert_model = experts[pred_country]
                expert_prob = expert_model.predict_proba(input_df)[0, 1]
                expert_model_used = True

            pooled_prob = model_pooled.predict_proba(input_df)[0, 1]

            # ===============================
            # MAIN RESULT (EXPERT OR POOLED)
            # ===============================
            final_prob = expert_prob if expert_model_used else pooled_prob
            model_name = f"Expert ({pred_country})" if expert_model_used else "Pooled Model"

            st.markdown("## 🎯 Prediction Result")

            st.markdown(
                f"""
                <div style="
                    background-color:#111827;
                    padding:25px;
                    border-radius:12px;
                    text-align:center;">
                    <h3 style="color:white;">Probability of Digital Financial Access</h3>
                    <h1 style="color:{get_color(final_prob)}; font-size:52px;">
                        {final_prob:.1%}
                    </h1>
                </div>
                """,
                unsafe_allow_html=True
            )

            colA, colB = st.columns(2)

            with colA:
                st.metric("Model Used", model_name)

            with colB:
                st.metric("Gating Confidence", f"{gating_conf:.1%}")

            # ===============================
            # INTERPRETATION
            # ===============================
            if final_prob >= 0.75:
                st.success("🟢 High likelihood of financial access")
            elif final_prob >= 0.50:
                st.info("🟠 Moderate likelihood of access")
            else:
                st.error("🔴 Low likelihood — barriers exist")

            # ===============================
            # COMPARISON SECTION (ADDED)
            # ===============================
            st.markdown("## 📊 Model Comparison")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Pooled Model")
                st.progress(pooled_prob)
                st.write(f"**{pooled_prob:.1%}**")

            with col2:
                st.markdown(f"### Expert Model ({pred_country})")

                if expert_prob is not None:
                    st.progress(expert_prob)
                    st.write(f"**{expert_prob:.1%}**")
                else:
                    st.warning("No expert model available")

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
