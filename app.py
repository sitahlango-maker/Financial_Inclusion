import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Digital Finance Access Predictor",
    page_icon="🌍",
    layout="wide"
)

# =========================================================
# STYLING
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3.1rem !important;
    font-weight: 700 !important;
    color: #0F2B46 !important;
}

.subtitle {
    font-size: 1.35rem;
    font-weight: 500;
    color: #2C5F7A;
    margin-bottom: 1.8rem;
}

.stApp {
    background: linear-gradient(135deg, #F8FAFC 0%, #E6F0F5 100%);
}

.main .block-container {
    background: rgba(255,255,255,0.95);
    padding: 2.2rem;
    border-radius: 24px;
    box-shadow: 0 15px 35px rgba(15,43,70,0.08);
    max-width: 1350px;
    margin: 1rem auto;
}

div[data-testid="stMetric"] {
    background: white;
    border-radius: 16px;
    padding: 1rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}

.stButton > button {
    background: linear-gradient(90deg, #1E88E5, #1565C0);
    color: white;
    font-weight: 600;
    border-radius: 12px;
    padding: 0.75rem 1.8rem;
    font-size: 1.05rem;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================
st.markdown("# 🌍 Digital Finance Access Predictor")
st.markdown(
    '<div class="subtitle">East Africa • Executive Intelligence Platform</div>',
    unsafe_allow_html=True
)

# =========================================================
# PATH
# =========================================================
MODEL_PATH = "."

# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_models():
    return {
        "feature_columns": joblib.load(os.path.join(MODEL_PATH, "feature_columns.joblib")),
        "pooled_model": joblib.load(os.path.join(MODEL_PATH, "pooled_model.joblib")),
        "harmonized_model": joblib.load(os.path.join(MODEL_PATH, "harmonized_model.joblib")),
        "expert_models": {
            "KEN": joblib.load(os.path.join(MODEL_PATH, "expert_model_KEN.joblib")),
            "TZA": joblib.load(os.path.join(MODEL_PATH, "expert_model_TZA.joblib")),
            "UGA": joblib.load(os.path.join(MODEL_PATH, "expert_model_UGA.joblib")),
        },
        "routing_model": joblib.load(os.path.join(MODEL_PATH, "routing_model.joblib")),
    }


models = load_models()

# =========================================================
# SESSION STATE
# =========================================================
if "page" not in st.session_state:
    st.session_state.page = "input"

if "results" not in st.session_state:
    st.session_state.results = None

# =========================================================
# CONFIG
# =========================================================
country_defaults = {
    "KEN": {"name": "Kenya"},
    "TZA": {"name": "Tanzania"},
    "UGA": {"name": "Uganda"},
}

# =========================================================
# DISPLAY NAMES
# =========================================================
DISPLAY_NAMES = {
    "gender": "Gender",
    "female": "Gender",
    "age": "Age",
    "educ": "Education",
    "inc_q": "Income Quintile",
    "urbanicity": "Residence",
    "internet_use": "Internet Use",
    "wgt": "Survey Weight",
    "reg_index": "Regulatory Index",
    "reg_cons_prot": "Consumer Protection",
    "reg_kyc_prop": "KYC Proportionality",
    "reg_entry_lim": "Entry Transaction Limits",
    "reg_max_lim": "Maximum Transaction Limits",
    "reg_agent_el": "Agent Eligibility",
    "num_providers": "Mobile Money Providers",
    "earliest_launch": "Mobile Money Launch Year",
    "mmpi_2023_Very high": "Mobile Money Prevalence",
    "country_code_KEN": "Country: Kenya",
    "country_code_TZA": "Country: Tanzania",
    "country_code_UGA": "Country: Uganda",
}


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def build_input_row(
    feature_columns,
    country,
    age,
    gender,
    residence,
    income,
    education,
    internet
):
    input_data = pd.DataFrame(
        np.zeros((1, len(feature_columns))),
        columns=feature_columns
    )

    values = {
        "age": age,
        "gender": 1 if gender == "Female" else 2,
        "female": 1 if gender == "Female" else 0,
        "inc_q": income,
        "educ": education,
        "urbanicity": 1 if residence == "Urban" else 2,
        "internet_use": 1 if internet == "Yes" else 0,
    }

    for col, value in values.items():
        if col in input_data.columns:
            input_data[col] = value

    country_col = f"country_code_{country}"

    if country_col in input_data.columns:
        input_data[country_col] = 1

    return input_data


def predict_all(input_data):
    pooled_prob = models["pooled_model"].predict_proba(input_data)[0, 1]
    harmonized_prob = models["harmonized_model"].predict_proba(input_data)[0, 1]

    expert_probs = {}

    for country_code, model in models["expert_models"].items():
        expert_probs[f"Expert_{country_code}"] = model.predict_proba(input_data)[0, 1]

    model_probs = pd.DataFrame({
        "Pooled": [pooled_prob],
        "Harmonized": [harmonized_prob],
        **expert_probs
    })

    router_input = pd.concat(
        [input_data, model_probs],
        axis=1
    )

    routed_model = models["routing_model"].predict(router_input)[0]

    if routed_model not in model_probs.columns:
        routed_model = model_probs.T[0].idxmax()

    final_prob = model_probs[routed_model].iloc[0]

    return model_probs, routed_model, final_prob


def get_feature_impact(selected_model_name):
    if selected_model_name == "Pooled":
        model = models["pooled_model"]
    elif selected_model_name == "Harmonized":
        model = models["harmonized_model"]
    elif selected_model_name.startswith("Expert_"):
        country_code = selected_model_name.replace("Expert_", "")
        model = models["expert_models"][country_code]
    else:
        model = models["pooled_model"]

    impact_df = pd.DataFrame({
        "Feature": models["feature_columns"],
        "Importance": model.feature_importances_
    })

    impact_df["Feature"] = impact_df["Feature"].replace(DISPLAY_NAMES)

    return impact_df.sort_values("Importance", ascending=False).head(10)


# =========================================================
# INPUT PAGE
# =========================================================
if st.session_state.page == "input":
    st.subheader("👤 Enter Client Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 80, 30)
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        residence = st.radio("Residence", ["Rural", "Urban"], horizontal=True)

    with col2:
        income = st.selectbox("Income Quintile", [1, 2, 3, 4, 5])
        education = st.selectbox(
            "Education Level",
            [0, 1, 2, 3, 4],
            format_func=lambda x: [
                "No Education",
                "Primary",
                "Secondary",
                "Tertiary",
                "Higher"
            ][x]
        )
        internet = st.radio("Internet Use", ["No", "Yes"], horizontal=True)

    with col3:
        country = st.selectbox(
            "Country",
            ["KEN", "TZA", "UGA"],
            format_func=lambda x: country_defaults[x]["name"]
        )

    if st.button("🔮 Generate Prediction", type="primary", use_container_width=True):
        input_data = build_input_row(
            models["feature_columns"],
            country,
            age,
            gender,
            residence,
            income,
            education,
            internet
        )

        model_probs, routed_model, final_prob = predict_all(input_data)

        st.session_state.results = {
            "model_probs": model_probs,
            "routed_model": routed_model,
            "final_prob": final_prob,
            "country": country,
            "age": age,
            "gender": gender,
            "residence": residence,
            "income": income,
            "education": education,
            "internet": internet,
        }

        st.session_state.page = "results"
        st.rerun()

# =========================================================
# RESULTS PAGE
# =========================================================
else:
    res = st.session_state.results
    model_probs = res["model_probs"]

    st.subheader("📊 Prediction Results")

    if st.button("🔄 New Prediction", type="secondary"):
        st.session_state.page = "input"
        st.rerun()

    pooled_prob = model_probs["Pooled"].iloc[0]
    harmonized_prob = model_probs["Harmonized"].iloc[0]

    expert_cols = [
        col for col in model_probs.columns
        if col.startswith("Expert_")
    ]

    best_expert = model_probs[expert_cols].idxmax(axis=1).iloc[0]
    best_expert_prob = model_probs[best_expert].iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🌐 Pooled Model", f"{pooled_prob:.1%}")
    col2.metric("🧩 Harmonized Model", f"{harmonized_prob:.1%}")
    col3.metric("🧠 Best Expert", f"{best_expert_prob:.1%}")
    col4.metric("🎯 Final MoE Score", f"{res['final_prob']:.1%}")

    st.caption(f"Router selected: **{res['routed_model']}**")

    col_a, col_b = st.columns(2)

    with col_a:
        fig1 = go.Figure()

        fig1.add_trace(go.Bar(
            x=model_probs.columns,
            y=model_probs.iloc[0] * 100,
            text=[f"{v:.1%}" for v in model_probs.iloc[0]],
            textposition="outside",
            marker_color=[
                "#90A4AE",
                "#FFB74D",
                "#1E88E5",
                "#43A047",
                "#8E24AA"
            ]
        ))

        fig1.update_layout(
            title="Model Prediction Comparison",
            yaxis_title="Predicted Probability (%)",
            height=380,
            margin=dict(t=45, b=40),
            yaxis=dict(range=[0, 100])
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        impact_df = get_feature_impact(res["routed_model"])

        fig2 = go.Figure()

        fig2.add_trace(go.Bar(
            y=impact_df["Feature"][::-1],
            x=impact_df["Importance"][::-1],
            orientation="h",
            text=[
                f"{v:.3f}"
                for v in impact_df["Importance"][::-1]
            ],
            textposition="outside",
            marker_color="#1E88E5"
        ))

        fig2.update_layout(
            title=f"Key Drivers - {res['routed_model']}",
            xaxis_title="Feature Importance",
            height=380,
            margin=dict(t=45, b=40)
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    st.subheader("📋 Model Probability Table")

    prob_table = model_probs.T.reset_index()
    prob_table.columns = ["Model", "Probability"]
    prob_table["Probability"] = prob_table["Probability"].map(
        lambda x: f"{x:.1%}"
    )

    st.dataframe(prob_table, use_container_width=True)


       st.subheader("📌 Feature Impact Table")
    st.dataframe(impact_df, use_container_width=True)

    prob = res["final_prob"]

    if prob >= 0.90:
        st.success("🟢 HIGH likelihood of digital financial inclusion")
    elif prob >= 0.70:
        st.warning("🟡 MODERATE likelihood of digital financial inclusion")
    else:
        st.error("🔴 LOW likelihood of digital financial inclusion")

st.markdown("---")
st.markdown(
    "**East Africa Digital Financial Inclusion Intelligence** | Research & Executive Analytics Platform © 2026"
)
