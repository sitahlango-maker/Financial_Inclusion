import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Digital Financial Inclusion Predictor",
    layout="wide"
)

st.title("🌍 Digital Financial Inclusion Predictor")
st.markdown("Mixture of Experts model for predicting access to digital financial services in East Africa.")

# ===============================
# LOAD MODELS
# ===============================
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

@st.cache_resource
def load_models():
    model_pooled = joblib.load(os.path.join(MODEL_DIR, "model_pooled.pkl"))
    gating_model = joblib.load(os.path.join(MODEL_DIR, "gating_model.pkl"))
    experts = joblib.load(os.path.join(MODEL_DIR, "experts.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
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

# Align features
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[feature_names]

# ===============================
# PREDICTION BUTTON
# ===============================
if st.button("🔮 Predict Digital Financial Access", type="primary"):

    with st.spinner("Running Mixture of Experts model..."):

        # -----------------------
        # Gating Model
        # -----------------------
        pred_country = gating_model.predict(input_df)[0]
        gating_conf = np.max(gating_model.predict_proba(input_df))

        # -----------------------
        # Route to expert or pooled
        # -----------------------
        if pred_country in experts and gating_conf > 0.4:
            final_model = experts[pred_country]
            model_name = f"Expert Model ({pred_country})"
            color = "#22D3EE"
        else:
            final_model = model_pooled
            model_name = "Pooled Model (Fallback)"
            color = "#F472B6"

        prob = final_model.predict_proba(input_df)[0, 1]

        # ===============================
        # RESULT DISPLAY
        # ===============================
        st.markdown("## 🎯 Prediction Result")

        colA, colB = st.columns([2, 1])

        with colA:
            st.markdown(f"""
            <div style="padding:20px; border-radius:12px; background:#111827;">
                <h3>Probability of Digital Financial Access</h3>
                <h1 style="color:{color}; font-size:48px;">
                    {prob:.1%}
                </h1>
            </div>
            """, unsafe_allow_html=True)

        with colB:
            st.metric("Model Used", model_name)
            st.metric("Gating Confidence", f"{gating_conf:.1%}")

        # -----------------------
        # INTERPRETATION
        # -----------------------
        if prob >= 0.75:
            st.success("🟢 High likelihood of digital financial access")
        elif prob >= 0.50:
            st.info("🔵 Moderate likelihood of access")
        else:
            st.warning("🟠 Lower likelihood — potential barriers exist")

        # ===============================
        # COMPARISON
        # ===============================
        st.markdown("## 🔍 Model Comparison")

        col1, col2 = st.columns(2)

        with col1:
            pooled_prob = model_pooled.predict_proba(input_df)[0, 1]
            st.write("### Pooled Model")
            st.progress(pooled_prob)
            st.write(f"{pooled_prob:.1%}")

        with col2:
            if pred_country in experts:
                expert_prob = experts[pred_country].predict_proba(input_df)[0, 1]
                st.write(f"### Expert ({pred_country})")
                st.progress(expert_prob)
                st.write(f"{expert_prob:.1%}")
            else:
                st.write("No expert model available")

        # ===============================
        # FEATURE IMPORTANCE
        # ===============================
        st.markdown("## 🏆 Feature Importance")

        try:
            importance = pd.Series(
                final_model.feature_importances_,
                index=feature_names
            ).sort_values(ascending=False).head(10)

            fig, ax = plt.subplots()
            sns.barplot(x=importance.values, y=importance.index, ax=ax)
            ax.set_title("Top 10 Features")
            st.pyplot(fig)

        except Exception:
            st.info("Feature importance not available for this model type.")

        # ===============================
        # SHAP EXPLANATION
        # ===============================
        st.markdown("## 📊 Model Explanation (SHAP)")

        try:
            explainer = shap.TreeExplainer(final_model)
            shap_values = explainer.shap_values(input_df)

            fig = plt.figure()
            shap.summary_plot(
                shap_values,
                input_df,
                plot_type="bar",
                show=False
            )
            st.pyplot(fig)

        except Exception:
            st.info("SHAP explanation not available for this model.")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.markdown(
    "<center>Digital Financial Inclusion Predictor • Mixture of Experts ML System</center>",
    unsafe_allow_html=True
)
