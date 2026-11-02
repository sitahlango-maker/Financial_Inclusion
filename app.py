import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

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
    '<div class="subtitle">Kenya • Tanzania • Uganda • Global Findex 2021</div>',
    unsafe_allow_html=True
)

# =========================================================
# LOAD MODELS (produced by train.py)
# =========================================================
MODEL_PATH = "models"


@st.cache_resource
def load_models():
    with open(os.path.join(MODEL_PATH, "model_info.json")) as f:
        info = json.load(f)

    base = {}
    for name in info["base_models"]:
        model = xgb.XGBClassifier()
        model.load_model(os.path.join(MODEL_PATH, f"{name}.json"))
        base[name] = model

    router = xgb.XGBClassifier()
    router.load_model(os.path.join(MODEL_PATH, "router.json"))
    return info, base, router


INFO, BASE_MODELS, ROUTER = load_models()
FEATURES = INFO["features"]

# =========================================================
# LABELS AND CODES (Global Findex 2021 codebook)
# =========================================================
COUNTRY_NAMES = {"KEN": "Kenya", "TZA": "Tanzania", "UGA": "Uganda"}
GENDER_CODES = {"Female": 1, "Male": 2}
RESIDENCE_CODES = {"Rural": 1, "Urban": 2}
EDUCATION_NAMES = {1: "Primary or less", 2: "Secondary", 3: "Tertiary or more"}
INCOME_NAMES = {1: "1 (poorest 20%)", 2: "2", 3: "3", 4: "4", 5: "5 (richest 20%)"}

MODEL_LABELS = {
    "Pooled": "Pooled (main estimate)",
    "Harmonized": "Harmonized",
    "Expert_KEN": "Kenya expert",
    "Expert_TZA": "Tanzania expert",
    "Expert_UGA": "Uganda expert",
    "MoE_CountryGate": "MoE: own-country expert",
    "MoE_Router": "MoE: learned router",
}

DISPLAY_NAMES = {
    "female": "Gender",
    "age": "Age",
    "educ": "Education",
    "inc_q": "Income quintile",
    "urbanicity": "Residence",
    "internet_use": "Internet use",
    "country_KEN": "Country: Kenya",
    "country_TZA": "Country: Tanzania",
    "country_UGA": "Country: Uganda",
}

DEFAULT_INPUTS = {
    "age": 30,
    "gender": "Female",
    "residence": "Rural",
    "income": 3,
    "education": 2,
    "internet": "No",
    "country": "KEN",
}

if "page" not in st.session_state:
    st.session_state.page = "input"
if "results" not in st.session_state:
    st.session_state.results = None
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = DEFAULT_INPUTS.copy()


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def build_input_row(inputs):
    row = {
        "female": GENDER_CODES[inputs["gender"]],
        "age": inputs["age"],
        "educ": inputs["education"],
        "inc_q": inputs["income"],
        "urbanicity": RESIDENCE_CODES[inputs["residence"]],
        "internet_use": 1 if inputs["internet"] == "Yes" else 0,
    }
    for code in COUNTRY_NAMES:
        row[f"country_{code}"] = 1 if inputs["country"] == code else 0
    return pd.DataFrame([row])[FEATURES]


def predict_all(input_data, country):
    probs = {name: float(m.predict_proba(input_data)[0, 1]) for name, m in BASE_MODELS.items()}

    router_input = pd.concat(
        [input_data, pd.DataFrame([probs])[INFO["base_models"]]], axis=1
    )[INFO["router_columns"]]
    routed = INFO["router_classes"][int(ROUTER.predict(router_input)[0])]

    probs["MoE_CountryGate"] = probs[f"Expert_{country}"]
    probs["MoE_Router"] = probs[routed]
    return probs, routed


def feature_contributions(model_name, input_data):
    """Per-profile contributions (TreeSHAP values computed by XGBoost)."""
    booster = BASE_MODELS[model_name].get_booster()
    dmatrix = xgb.DMatrix(input_data, feature_names=FEATURES)
    contribs = booster.predict(dmatrix, pred_contribs=True)[0][:-1]

    df = pd.DataFrame({"Feature": FEATURES, "Contribution": contribs})
    df["Feature"] = df["Feature"].replace(DISPLAY_NAMES)
    df["Absolute"] = df["Contribution"].abs()
    df = df[df["Absolute"] > 0].sort_values("Absolute", ascending=False)
    return df.drop(columns="Absolute")


def likelihood_message(prob):
    if prob >= 0.5:
        return "success", "🟢 This profile is LIKELY to have a digital financial account."
    return "error", "🔴 This profile is UNLIKELY to have a digital financial account."


# =========================================================
# INPUT PAGE
# =========================================================
if st.session_state.page == "input":
    st.subheader("👤 Enter a Profile")
    saved = st.session_state.last_inputs

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 15, 90, saved["age"])
        gender = st.radio("Gender", ["Female", "Male"],
                          index=["Female", "Male"].index(saved["gender"]), horizontal=True)
        residence = st.radio("Residence", ["Rural", "Urban"],
                             index=["Rural", "Urban"].index(saved["residence"]), horizontal=True)

    with col2:
        income = st.selectbox("Income quintile (within country)", [1, 2, 3, 4, 5],
                              index=[1, 2, 3, 4, 5].index(saved["income"]),
                              format_func=lambda x: INCOME_NAMES[x])
        education = st.selectbox("Education level", [1, 2, 3],
                                 index=[1, 2, 3].index(saved["education"]),
                                 format_func=lambda x: EDUCATION_NAMES[x])
        internet = st.radio("Uses the internet", ["No", "Yes"],
                            index=["No", "Yes"].index(saved["internet"]), horizontal=True)

    with col3:
        country = st.selectbox("Country", list(COUNTRY_NAMES),
                               index=list(COUNTRY_NAMES).index(saved["country"]),
                               format_func=lambda x: COUNTRY_NAMES[x])

    if st.button("🔮 Generate Prediction", type="primary", use_container_width=True):
        inputs = {"age": age, "gender": gender, "residence": residence, "income": income,
                  "education": education, "internet": internet, "country": country}
        st.session_state.last_inputs = inputs

        input_data = build_input_row(inputs)
        probs, routed = predict_all(input_data, country)

        st.session_state.results = {"probs": probs, "routed": routed,
                                    "input_data": input_data, "inputs": inputs}
        st.session_state.page = "results"
        st.rerun()

# =========================================================
# RESULTS PAGE
# =========================================================
else:
    res = st.session_state.results
    probs, inputs = res["probs"], res["inputs"]
    country = inputs["country"]
    base_rate = INFO["country_base_rate"][country]

    st.subheader("📊 Prediction Results")
    st.info(
        f"Profile: {COUNTRY_NAMES[country]} | Age {inputs['age']} | {inputs['gender']} | "
        f"{inputs['residence']} | Income quintile {inputs['income']} | "
        f"{EDUCATION_NAMES[inputs['education']]} | Internet: {inputs['internet']}"
    )

    btn1, btn2 = st.columns(2)
    with btn1:
        if st.button("⬅️ Back / Edit Profile", use_container_width=True):
            st.session_state.page = "input"
            st.rerun()
    with btn2:
        if st.button("🔄 New Prediction", use_container_width=True):
            st.session_state.last_inputs = DEFAULT_INPUTS.copy()
            st.session_state.results = None
            st.session_state.page = "input"
            st.rerun()

    main_prob = probs["Pooled"]
    c1, c2, c3 = st.columns(3)
    c1.metric("🎯 Estimated probability (Pooled model)", f"{main_prob:.1%}")
    c2.metric(f"📍 Average in {COUNTRY_NAMES[country]}", f"{base_rate:.1%}",
              help="Share of survey respondents in this country with a digital account.")
    c3.metric("🧠 Own-country expert", f"{probs[f'Expert_{country}']:.1%}")

    st.caption(
        "The Pooled model is the main estimate because it was among the strongest in testing "
        "(see results/ in the repository). The other models are shown for comparison. "
        f"The experimental learned router selected: **{MODEL_LABELS[res['routed']]}**."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        order = ["Pooled", "Harmonized", f"Expert_{country}", "MoE_Router"]
        fig1 = go.Figure(go.Bar(
            x=[MODEL_LABELS[m] for m in order],
            y=[probs[m] * 100 for m in order],
            text=[f"{probs[m]:.1%}" for m in order],
            textposition="outside",
            marker_color=["#1E88E5", "#FFB74D", "#43A047", "#8E24AA"],
        ))
        fig1.add_hline(y=base_rate * 100, line_dash="dot",
                       annotation_text=f"{COUNTRY_NAMES[country]} average")
        fig1.update_layout(title="Model Estimates for This Profile",
                           yaxis_title="Probability (%)", yaxis=dict(range=[0, 105]),
                           height=380, margin=dict(t=45, b=40))
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        impact_df = feature_contributions("Pooled", res["input_data"])
        fig2 = go.Figure(go.Bar(
            y=impact_df["Feature"][::-1],
            x=impact_df["Contribution"][::-1],
            orientation="h",
            text=[f"{v:+.2f}" for v in impact_df["Contribution"][::-1]],
            textposition="outside",
            marker_color=["#43A047" if v > 0 else "#E53935"
                          for v in impact_df["Contribution"][::-1]],
        ))
        fig2.update_layout(title="Why? Feature Contributions (Pooled model)",
                           xaxis_title="Push towards (+) or away from (−) having an account "
                                       "(log-odds)",
                           height=380, margin=dict(t=45, b=40))
        st.plotly_chart(fig2, use_container_width=True)

    kind, message = likelihood_message(main_prob)
    getattr(st, kind)(message)

    st.markdown("---")
    st.subheader("📋 All Model Estimates")
    table = pd.DataFrame({
        "Model": [MODEL_LABELS[m] for m in MODEL_LABELS],
        "Probability": [f"{probs[m]:.1%}" for m in MODEL_LABELS],
    })
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("📌 Feature Contribution Table (Pooled model)")
    st.caption("Positive values push the estimate up; negative values push it down. "
               "Values are TreeSHAP contributions in log-odds.")
    st.dataframe(impact_df.round(3), use_container_width=True, hide_index=True)

    st.info(
        "These are statistical estimates from survey data (Global Findex 2021, about 1,000 "
        "respondents per country). They describe patterns, not individuals, and should "
        "support, not replace, human judgement."
    )

st.markdown("---")
st.markdown(
    "**Digital Finance Access Predictor** | MSc research project by Winnie MejaSitah Lang'o © 2026"
)
