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

# ====================== LIGHT EXECUTIVE THEME ======================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #F5F8FC 0%, #EEF3F9 50%, #E9EEF6 100%);
    color: #1F2A44;
}

.main .block-container {
    background: rgba(255, 255, 255, 0.75);
    border: 1px solid rgba(31, 42, 68, 0.08);
    border-radius: 18px;
    padding: 2.5rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
    backdrop-filter: blur(10px);
}

h1 {
    color: #0F2D4A;
    font-weight: 700;
}

h2, h3 {
    color: #1F4B7A;
    font-weight: 600;
}

p, label {
    color: #2F3B52 !important;
}

.stButton>button {
    background: linear-gradient(90deg, #14B8A6, #0EA5A4);
    color: white;
    font-weight: 600;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #0EA5A4, #2DD4BF);
    transform: translateY(-1px);
}

div[data-testid="metric-container"] {
    background-color: rgba(255, 255, 255, 0.9);
    border: 1px solid rgba(15, 45, 74, 0.08);
    padding: 14px;
    border-radius: 14px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
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

# ====================== FEATURE IMPORTANCE ======================
def get_feature_importance(model, feature_names):

    if hasattr(model, "feature_importances_"):
        imp = np.array(model.feature_importances_)
    elif hasattr(model, "coef_"):
        imp = np.abs(np.array(model.coef_).flatten())
    else:
        return None

    n = min(len(feature_names), len(imp))

    return pd.DataFrame({
        "feature": feature_names[:n],
        "importance": imp[:n]
    }).sort_values("importance", ascending=False)


def plot_feature_importance(model):

    fi = get_feature_importance(model, models["feature_names"])

    if fi is None or fi.empty:
        st.info("Feature importance not available.")
        return

    fi = fi.head(10)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=fi["importance"],
        y=fi["feature"],
        orientation="h",
        marker=dict(color=fi["importance"], colorscale="Teal")
    ))

    fig.update_layout(
        title="Key Drivers of Digital Financial Access",
        height=450,
        template="plotly_white",
        margin=dict(l=120, r=30, t=50, b=30)
    )

    st.plotly_chart(fig, use_container_width=True)

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
    country_input = st.selectbox("Country", ["KEN (Kenya)", "TZA (Tanzania)", "UGA (Uganda)"])
    country_code = country_input.split()[0]

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
    "KEN": {"reg_index": 0.85, "mmpi_2023": 0.75, "num_providers": 5, "earliest_launch": 2010},
    "TZA": {"reg_index": 0.72, "mmpi_2023": 0.61, "num_providers": 4, "earliest_launch": 2014},
    "UGA": {"reg_index": 0.78, "mmpi_2023": 0.68, "num_providers": 4, "earliest_launch": 2012}
}

input_dict.update(country_features.get(country_code, {}))

input_df = pd.DataFrame([input_dict])

if models:
    input_df = input_df.reindex(columns=models["feature_names"], fill_value=0)

# ====================== MODEL COMPARISON ======================
def predict_models(X_input):

    X_input = X_input.reindex(columns=models["feature_names"], fill_value=0)

    # Pooled model
    pooled_prob = models["pooled"].predict_proba(X_input)[0, 1]

    # Gating model
    pred_country = models["gating"].predict(X_input)[0]

    expert_prob = pooled_prob
    model_used = "Pooled"

    if pred_country in models["experts"]:
        try:
            expert = models["experts"][pred_country]

            sample = X_input.copy()

            if hasattr(expert, "feature_names_in_"):
                sample = sample.reindex(columns=expert.feature_names_in_, fill_value=0)

            expert_prob = expert.predict_proba(sample)[0, 1]
            model_used = f"Expert ({pred_country})"

        except:
            expert_prob = pooled_prob
            model_used = "Pooled (fallback)"

    return pooled_prob, expert_prob, model_used

# ====================== PREDICTION ======================
if st.button("🔮 Predict Digital Inclusion", type="primary"):

    if models:

        try:
            pooled_prob, expert_prob, model_used = predict_models(input_df)

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
                st.success("🟢 High likelihood of digital inclusion")
            elif final_prob > 0.5:
                st.warning("🟡 Moderate likelihood")
            else:
                st.error("🔴 Low likelihood")

            st.markdown("### 📊 Key Drivers")
            plot_feature_importance(models["pooled"])

        except Exception as e:
            st.error(f"Prediction error: {e}")

    else:
        st.error("Models not loaded")
