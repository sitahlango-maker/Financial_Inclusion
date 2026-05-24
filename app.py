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

# Theme
st.markdown("""
<style>

/* ===== BACKGROUND (INSTITUTIONAL FINANCE STYLE) ===== */
.stApp {
    background: linear-gradient(135deg, #0B1F3A 0%, #102A43 50%, #0F172A 100%);
    color: #E5E7EB;
}

/* ===== MAIN GLASS CONTAINER ===== */
.main .block-container {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 2.5rem;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.45);
    backdrop-filter: blur(12px);
}

/* ===== HEADINGS ===== */
h1 {
    color: #E6F1FF;
    font-weight: 700;
    letter-spacing: 0.5px;
}

h2, h3 {
    color: #A7C7E7;
    font-weight: 600;
}

/* ===== TEXT ===== */
p, label, .stMarkdown {
    color: #D1D5DB !important;
}

/* ===== BUTTONS (TEAL FINANCE ACCENT) ===== */
.stButton>button {
    background: linear-gradient(90deg, #0EA5A4, #14B8A6);
    color: #0B1F3A;
    font-weight: 600;
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1.2rem;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #14B8A6, #2DD4BF);
    transform: translateY(-1px);
}

/* ===== METRICS CARDS ===== */
div[data-testid="metric-container"] {
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 12px;
    border-radius: 12px;
}

/* ===== INPUT FIELDS ===== */
.stSelectbox, .stRadio, .stSlider {
    background-color: rgba(255, 255, 255, 0.04);
    border-radius: 10px;
}

/* ===== STATUS COLORS ===== */
.stSuccess { color: #34D399 !important; }
.stWarning { color: #FBBF24 !important; }
.stError { color: #F87171 !important; }

</style>
""", unsafe_allow_html=True)

# ====================== TITLE ======================
st.title("💳 Digital Finance Access Predictor")
st.markdown("### East Africa • Kenya | Tanzania | Uganda")

# ====================== LOAD MODELS ======================
@st.cache_resource
def load_models():
    model_dir = "trained_models"

    try:
        models = {
            "pooled": joblib.load(f"{model_dir}/model_pooled.joblib"),
            "experts": joblib.load(f"{model_dir}/experts.joblib"),
            "gating": joblib.load(f"{model_dir}/gating_model.joblib"),
            "feature_names": joblib.load(f"{model_dir}/feature_names.joblib"),
            "medians": joblib.load(f"{model_dir}/medians.joblib")
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
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
    else:
        return None

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": imp
    }).sort_values("importance", ascending=False)

    return df


def plot_feature_importance(model):
    fi = get_feature_importance(model, models["feature_names"])

    if fi is None:
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
        template="plotly_white",
        height=450,
        margin=dict(l=120, r=30, t=50, b=30)
    )

    st.plotly_chart(fig, use_container_width=True)

# ====================== INPUT UI ======================
st.subheader("👤 User Profile")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 80, 32)
    female = st.radio("Gender", ["Male", "Female"], horizontal=True)
    urbanicity = st.radio("Location", ["Rural", "Urban"], horizontal=True)

with col2:
    inc_q = st.selectbox("Income Quintile", [1, 2, 3, 4, 5])
    educ = st.selectbox("Education Level", [0, 1, 2, 3, 4],
                        format_func=lambda x: ["None", "Primary", "Secondary", "Tertiary", "Higher"][x])
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

# ====================== GATING PREDICTION ======================
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
if st.button("🔮 Predict", type="primary"):

    if models:

        try:
            prob = predict_with_gating(input_df)[0]

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Digital Account Probability", f"{prob*100:.1f}%")

            with col2:
                st.metric("Risk Category",
                          "High" if prob > 0.75 else "Medium" if prob > 0.5 else "Low")

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
