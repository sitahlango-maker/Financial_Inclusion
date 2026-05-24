import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Digital Finance Access Predictor",
    page_icon="💰",
    layout="wide"
)

# ====================== STYLING ======================
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #4C1D95 0%, #6B46C1 50%, #9F7AEA 100%);
        color: white;
    }
    .main .block-container {
        background: rgba(255, 255, 255, 0.13);
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
    }
    h1, h2, h3 { color: #E0BBFF; }
    .stButton>button {
        background: linear-gradient(90deg, #C4B5FD, #A78BFA);
        color: #2D1B69;
        font-weight: bold;
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💰 Digital Finance Access Predictor")
st.markdown("### East Africa • Kenya | Tanzania | Uganda")

# ====================== LOAD MODELS ======================
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

        st.success("✅ All models loaded successfully!")
        return models

    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None


models = load_models()

# ====================== FEATURE IMPORTANCE ======================
def get_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return None

    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    return fi


def plot_feature_importance(model):
    fi = get_feature_importance(model, models["feature_names"])

    if fi is None:
        st.info("Feature importance not available for this model.")
        return

    fi = fi.head(10)

    colors = px.colors.qualitative.Set3

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=fi["importance"],
        y=fi["feature"],
        orientation="h",
        marker=dict(color=colors[:len(fi)])
    ))

    fig.update_layout(
        title="Key Drivers of Digital Financial Access",
        xaxis_title="Importance",
        yaxis_title="Features",
        height=450,
        template="plotly_white",
        font=dict(size=13),
        margin=dict(l=120, r=30, t=60, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)


# ====================== INPUT FORM ======================
st.subheader("👤 Enter User Profile")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 80, 32)
    female = st.radio("Gender", ["Male", "Female"], horizontal=True)
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
    country_input = st.selectbox("Country", ["KEN (Kenya)", "TZA (Tanzania)", "UGA (Uganda)"])
    country_code = country_input.split()[0]


# ====================== INPUT PREPARATION ======================
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
def predict_with_gating(X_input, return_routing_info=True):
    if not isinstance(X_input, pd.DataFrame):
        X_input = pd.DataFrame(X_input)

    X_input = X_input.reindex(columns=models["feature_names"], fill_value=0)

    predicted_country = models["gating"].predict(X_input)

    results = []
    routing_info = []

    for i, country in enumerate(predicted_country):
        sample = X_input.iloc[[i]]

        if country in models["experts"]:
            try:
                expert = models["experts"][country]

                if hasattr(expert, "feature_names_in_"):
                    sample = sample.reindex(columns=expert.feature_names_in_, fill_value=0)

                prob = expert.predict_proba(sample)[0, 1]
                model_used = f"Expert_{country}"

            except:
                prob = models["pooled"].predict_proba(sample)[0, 1]
                model_used = "Pooled (fallback)"
        else:
            prob = models["pooled"].predict_proba(sample)[0, 1]
            model_used = "Pooled"

        results.append(prob)
        routing_info.append(model_used)

    if return_routing_info:
        return np.array(results), routing_info

    return np.array(results)


# ====================== PREDICTION UI ======================
if st.button("🔮 Predict Digital Account Ownership", type="primary"):

    if models:

        try:
            probs, routing_info = predict_with_gating(input_df, True)

            prob = probs[0]
            model_used = routing_info[0]

            st.success("### Prediction Result")

            col_a, col_b = st.columns(2)

            with col_a:
                st.metric("Probability of Digital Account", f"{prob*100:.1f}%")

            with col_b:
                st.metric("Model Used", model_used)

            if prob >= 0.75:
                st.success("🟢 High Likelihood of Digital Inclusion")
            elif prob >= 0.5:
                st.warning("🟡 Moderate Likelihood")
            else:
                st.error("🔴 Low Likelihood")

            # ================= FEATURE IMPORTANCE =================
            st.markdown("### 📊 Key Drivers of Prediction")
            plot_feature_importance(models["pooled"])

        except Exception as e:
            st.error(f"Prediction error: {e}")

    else:
        st.error("Models not loaded. Check file paths.")
