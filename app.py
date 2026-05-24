import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Digital Finance Access Predictor",
    page_icon="💳",
    layout="wide"
)

# ====================== LIGHT SWIFT EXECUTIVE THEME ======================
st.markdown("""
<style>

/* ===================== LIGHT BACKGROUND ===================== */
.stApp {
    background: linear-gradient(135deg, #F5F8FC 0%, #EEF3F9 50%, #E9EEF6 100%);
    color: #1F2A44;
}

/* ===================== MAIN CONTAINER ===================== */
.main .block-container {
    background: rgba(255, 255, 255, 0.75);
    border: 1px solid rgba(31, 42, 68, 0.08);
    border-radius: 18px;
    padding: 2.5rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
    backdrop-filter: blur(10px);
}

/* ===================== HEADINGS ===================== */
h1 {
    color: #0F2D4A;
    font-weight: 700;
    letter-spacing: 0.3px;
}

h2, h3 {
    color: #1F4B7A;
    font-weight: 600;
}

/* ===================== TEXT ===================== */
p, label, .stMarkdown {
    color: #2F3B52 !important;
}

/* ===================== BUTTONS ===================== */
.stButton>button {
    background: linear-gradient(90deg, #14B8A6, #0EA5A4);
    color: white;
    font-weight: 600;
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1.2rem;
    box-shadow: 0 4px 12px rgba(20, 184, 166, 0.25);
}

.stButton>button:hover {
    background: linear-gradient(90deg, #0EA5A4, #2DD4BF);
    transform: translateY(-1px);
}

/* ===================== METRIC CARDS ===================== */
div[data-testid="metric-container"] {
    background-color: rgba(255, 255, 255, 0.9);
    border: 1px solid rgba(15, 45, 74, 0.08);
    padding: 14px;
    border-radius: 14px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

/* ===================== INPUT ELEMENTS ===================== */
.stSelectbox, .stRadio, .stSlider {
    background-color: rgba(255, 255, 255, 0.9);
    border-radius: 10px;
}

/* ===================== STATUS COLORS ===================== */
.stSuccess { color: #16A34A !important; }
.stWarning { color: #D97706 !important; }
.stError { color: #DC2626 !important; }

</style>
""", unsafe_allow_html=True)

# ====================== TITLE ======================
st.title("💳 Digital Finance Access Predictor")
st.markdown("### East Africa • Kenya | Tanzania | Uganda")

# ====================== LOAD MODELS ======================
@st.cache_resource
def load_models():
    try:
        model_dir = "trained_models"

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

    # 🔒 SAFETY: align lengths to avoid crash
    n = min(len(feature_names), len(imp))

    fi = pd.DataFrame({
        "feature": feature_names[:n],
        "importance": imp[:n]
    }).sort_values("importance", ascending=False)

    return fi


def plot_feature_importance(model):

fi = get_feature_importance(model, model.feature_names_in_)

    if fi is None or fi.empty:
        st.info("Feature importance not available.")
        return

    fi = fi.head(10)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=fi["importance"],
        y=fi["feature"],
        orientation="h",
        marker=dict(
            color=fi["importance"],
            colorscale="Teal"
        )
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

# ====================== GATING MODEL ======================
def predict_with_gating(X_input):
    X_input = X_input.reindex(columns=models["feature_names"], fill_value=0)

    pred_country = models["gating"].predict(X_input)

    results = []

    for i, country in enumerate(pred_country):
        sample = X_input.iloc[[i]]

        if country in models["experts"]:
            try:
                expert = models["experts"][country]

                if hasattr(expert, "feature_names_in_"):
                    sample = sample.reindex(columns=expert.feature_names_in_, fill_value=0)

                prob = expert.predict_proba(sample)[0, 1]
            except:
                prob = models["pooled"].predict_proba(sample)[0, 1]
        else:
            prob = models["pooled"].predict_proba(sample)[0, 1]

        results.append(prob)

    return np.array(results)

# ====================== PREDICTION ======================
if st.button("🔮 Predict Digital Inclusion", type="primary"):

    if models:

        try:
            prob = predict_with_gating(input_df)[0]

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Digital Account Probability", f"{prob*100:.1f}%")

            with col2:
                st.metric(
                    "Risk Level",
                    "High" if prob > 0.75 else "Medium" if prob > 0.5 else "Low"
                )

            if prob > 0.75:
                st.success("High likelihood of digital financial inclusion")
            elif prob > 0.5:
                st.warning("Moderate likelihood")
            else:
                st.error("Low likelihood")

            # ================= FEATURE DRIVERS =================
            st.markdown("### 📊 Key Drivers")
            plot_feature_importance(models["pooled"])

        except Exception as e:
            st.error(f"Prediction error: {e}")

    else:
        st.error("Models not loaded")
