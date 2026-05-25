import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# PAGE CONFIG & STYLING
# =========================================================
st.set_page_config(
    page_title="Digital Finance Access Predictor",
    page_icon="🌍💸",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #4C1D95 0%, #6B46C1 50%, #9F7AEA 100%); color: white; }
    .main .block-container { background: rgba(255,255,255,0.13); border-radius: 20px; padding: 2rem; }
    .stButton>button { background: linear-gradient(90deg, #4F8BF9, #7A5CFA); color: white; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🌍💸 Digital Finance Access Predictor")
st.markdown("### East Africa Digital Financial Inclusion Intelligence System")

# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_models():
    try:
        model_dir = "trained_models"
        models = {
            "pooled": joblib.load(f"{model_dir}/model_pooled.joblib"),
            "experts": joblib.load(f"{model_dir}/experts.joblib"),
            "gating": joblib.load(f"{model_dir}/gating_model.joblib"),
            "feature_names": joblib.load(f"{model_dir}/feature_names.joblib"),
            "medians": joblib.load(f"{model_dir}/medians.joblib")
        }
        st.success("✅ All models loaded successfully")
        return models
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

models = load_models()

# =========================================================
# USER INPUT
# =========================================================
st.subheader("👤 Enter User Profile")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 80, 32)
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    location = st.radio("Residence", ["Rural", "Urban"], horizontal=True)

with col2:
    income = st.selectbox("Income Quintile", [1, 2, 3, 4, 5])
    education = st.selectbox("Education Level", [0,1,2,3,4],
                             format_func=lambda x: ["No Education","Primary","Secondary","Tertiary","Higher"][x])
    internet = st.radio("Internet Use", ["No", "Yes"], horizontal=True)

with col3:
    country = st.selectbox("Country", ["KEN", "TZA", "UGA"],
                           format_func=lambda x: {"KEN":"Kenya", "TZA":"Tanzania", "UGA":"Uganda"}[x])

# =========================================================
# BUILD INPUT
# =========================================================
input_row = {
    "female": 1 if gender == "Female" else 0,
    "age": age,
    "educ": education,
    "inc_q": income,
    "urbanicity": 1 if location == "Urban" else 0,
    "internet_use": 1 if internet == "Yes" else 0,
    "dig_account": 0,
    "anydigpayment": 0,
    "wgt": 1.0,
    # Country-level features
    "reg_index": 0.85 if country == "KEN" else 0.72 if country == "TZA" else 0.78,
    "reg_cons_prot": 0.90 if country == "KEN" else 0.80 if country == "TZA" else 0.76,
    "reg_kyc_prop": 0.92 if country == "KEN" else 0.78 if country == "TZA" else 0.74,
    "reg_entry_lim": 0.85 if country == "KEN" else 0.75 if country == "TZA" else 0.73,
    "reg_max_lim": 0.88 if country == "KEN" else 0.77 if country == "TZA" else 0.72,
    "reg_agent_el": 0.96 if country == "KEN" else 0.84 if country == "TZA" else 0.80,
    "num_providers": 5 if country == "KEN" else 4 if country == "TZA" else 4,
    "earliest_launch": 2010 if country == "KEN" else 2014 if country == "TZA" else 2012,
}

df_input = pd.DataFrame([input_row])
df_input = df_input.reindex(columns=models["feature_names"], fill_value=0)

# =========================================================
# PREDICTION
# =========================================================
if st.button("🔮 Predict Digital Inclusion", type="primary"):
    try:
        probs, routing_info = predict_with_gating(df_input, return_routing_info=True)
        final_prob = probs[0]
        routed_model = routing_info[0]

        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("🌐 Pooled Model", f"{probs[0]:.1%}")
        col2.metric("🧠 Expert Model", f"{probs[0]:.1%}")
        col3.metric("🧭 Routed To", routed_model)

        st.progress(float(final_prob))
        st.subheader(f"Final Probability: **{final_prob:.1%}**")

        if final_prob >= 0.75:
            st.success("🟢 High likelihood of digital financial inclusion")
        elif final_prob >= 0.50:
            st.warning("🟡 Moderate likelihood")
        else:
            st.error("🔴 Low likelihood of digital financial inclusion")

        # =========================================================
        # FEATURE IMPORTANCE GRAPH
        # =========================================================
        st.markdown("### 🔍 Top Features Impacting This Prediction")
        
        if routed_model.startswith("Expert"):
            model_name = routed_model.replace("Expert_", "")
            model = models["experts"].get(model_name)
        else:
            model = models["pooled"]
            model_name = "Pooled"

        if hasattr(model, 'feature_importances_'):
            if hasattr(model, 'feature_names_in_'):
                feat_names = model.feature_names_in_
            else:
                feat_names = models["feature_names"]
            
            imp_series = pd.Series(model.feature_importances_, index=feat_names).sort_values(ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=imp_series.values, y=imp_series.index, ax=ax, palette="viridis")
            ax.set_title(f"Top 10 Features - {model_name} Model", fontsize=14)
            ax.set_xlabel("Importance Score")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("Digital Finance Access Predictor • Mixture-of-Experts Architecture • East Africa")
