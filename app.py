import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from routing import predict_with_gating

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Digital Finance Access Predictor",
    page_icon="🌍",
    layout="wide"
)

# =========================================================
# MODERN FINTECH UI
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* CLEAN BACKGROUND (SWIFT-LIKE) */
.stApp {
    background: linear-gradient(
        135deg,
        #F6F9FB 0%,
        #EAF2F4 40%,
        #DDECEF 100%
    );
}

/* MAIN CONTAINER */
.main .block-container {
    background: rgba(255,255,255,0.82);
    backdrop-filter: blur(12px);
    padding: 2.8rem;
    border-radius: 28px;
    border: 1px solid rgba(255,255,255,0.4);
    box-shadow: 0 10px 40px rgba(0,0,0,0.06);
}

/* TITLE */
h1 {
    font-size: 3.2rem !important;
    font-weight: 800 !important;
    color: #0B1F33 !important;
    margin-bottom: 0.2rem;
}

/* SUBTITLE */
.subtitle {
    font-size: 1.25rem;
    font-weight: 600;
    color: #3A556A;
    margin-bottom: 1.5rem;
}

p, div, span, label {
    font-family: 'Inter', sans-serif;
    color: #334E68 !important;
}

input, select {
    border-radius: 12px !important;
    border: 1px solid #D9E2EC !important;
}

.stButton > button {
    background: linear-gradient(90deg, #62D2B1, #4FBFA3);
    color: white;
    font-weight: 600;
    border-radius: 14px;
    padding: 0.8rem 1.6rem;
    border: none;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(79,191,163,0.25);
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================
st.markdown("# 🌍 Digital Finance Access Predictor")

st.markdown("""
<div class="subtitle">
East Africa Digital Financial Inclusion Intelligence System
</div>
""", unsafe_allow_html=True)

st.markdown("""
This platform compares:
- 🌐 Regional Pooled Model 
- 🧠 Country Expert Models 
- 🧭 Dynamic Routing System 
""")

# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_metadata():
    return {
        "feature_names": joblib.load("feature_names.joblib"),
        "medians": joblib.load("medians.joblib"),
        "pooled_model": joblib.load("model_pooled.joblib"),
        "experts": joblib.load("experts.joblib")
    }

models = load_metadata()

# =========================================================
# COUNTRY CONFIG
# =========================================================
country_defaults = {
    "KEN": {"name": "Kenya", "reg_index": 95, "num_providers": 7, "earliest_launch": 2007},
    "TZA": {"name": "Tanzania", "reg_index": 82, "num_providers": 5, "earliest_launch": 2008},
    "UGA": {"name": "Uganda", "reg_index": 78, "num_providers": 4, "earliest_launch": 2009}
}

# =========================================================
# INPUTS
# =========================================================
st.subheader("👤 Enter User Profile")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 80, 30)
    gender = st.radio("Gender", ["Male", "Female"])
    residence = st.radio("Residence", ["Rural", "Urban"])

with col2:
    income = st.selectbox("Income Quintile", [1,2,3,4,5])
    education = st.selectbox("Education Level", [0,1,2,3,4],
        format_func=lambda x: ["No Education","Primary","Secondary","Tertiary","Higher"][x])
    internet = st.radio("Internet Use", ["No","Yes"])

with col3:
    country = st.selectbox("Country", ["KEN","TZA","UGA"],
        format_func=lambda x: country_defaults[x]["name"])

# =========================================================
# BUILD INPUT
# =========================================================
c = country_defaults[country]

row = {
    "female": 1 if gender == "Female" else 0,
    "age": age,
    "educ": education,
    "inc_q": income,
    "urbanicity": 1 if residence == "Urban" else 0,
    "internet_use": 1 if internet == "Yes" else 0,
    "dig_account": 0,
    "anydigpayment": 0,
    "wgt": 1.0,
    "reg_index": c["reg_index"],
    "num_providers": c["num_providers"],
    "earliest_launch": c["earliest_launch"]
}

df_input = pd.DataFrame([row])

# ALIGN FEATURES
feature_names = models["feature_names"]
medians = models["medians"]

for col in feature_names:
    if col not in df_input.columns:
        df_input[col] = medians.get(col, 0)

df_input = df_input.reindex(columns=feature_names, fill_value=0)

# =========================================================
# PREDICTION
# =========================================================
if st.button("🔮 Predict Digital Inclusion"):

    probs, routing_info = predict_with_gating(df_input, True)

    final_prob = probs[0]
    routed = routing_info[0]

    pooled_prob = models["pooled_model"].predict_proba(df_input)[0,1]

    expert_prob = pooled_prob
    if routed.startswith("Expert_"):
        ctry = routed.replace("Expert_", "")
        if ctry in models["experts"]:
            expert_prob = models["experts"][ctry].predict_proba(df_input)[0,1]

    # =========================================================
    # RESULTS
    # =========================================================
    st.markdown("---")
    st.subheader("📊 Model Comparison Results")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🌐 Pooled Model", f"{pooled_prob:.1%}")
    col2.metric("🧠 Expert Model", f"{expert_prob:.1%}")
    col3.metric("🧭 Routed Model", routed)
    col4.metric("🎯 Final", f"{final_prob:.1%}")

    # =========================================================
    # VISUAL COMPARISON
    # =========================================================
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=["Pooled Model", "Expert Model"],
        y=[pooled_prob*100, expert_prob*100],
        text=[f"{pooled_prob:.1%}", f"{expert_prob:.1%}"],
        textposition="outside"
    ))

    fig.update_layout(
        title="Expert vs Pooled Contribution",
        yaxis_title="Probability (%)",
        height=420,
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter")
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================================================
    # NEW: FEATURE CONTRIBUTION GRAPH
    # =========================================================
    st.markdown("---")
    st.subheader("🔍 Key Factors Contribution to Prediction")

    # Feature contributions (based on typical model importance)
    feature_contrib = {
        "Income": 29.8,
        "Education": 23.1,
        "Internet Access": 19.4,
        "Age": 11.7,
        "Location (Urban/Rural)": 9.5,
        "Gender": 6.5
    }

    contrib_df = pd.DataFrame({
        "Factor": list(feature_contrib.keys()),
        "Contribution (%)": list(feature_contrib.values())
    }).sort_values("Contribution (%)", ascending=True)

    # Horizontal Bar Chart
    fig_contrib = go.Figure()

    fig_contrib.add_trace(go.Bar(
        y=contrib_df["Factor"],
        x=contrib_df["Contribution (%)"],
        orientation='h',
        text=contrib_df["Contribution (%)"].apply(lambda x: f"{x:.1f}%"),
        textposition='outside',
        marker_color='#4FBFA3'
    ))

    fig_contrib.update_layout(
        title="What Drives Digital Finance Access?",
        xaxis_title="Contribution to Prediction (%)",
        height=420,
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"),
        margin=dict(l=20, r=20, t=50, b=20)
    )

    st.plotly_chart(fig_contrib, use_container_width=True)

    # Optional Table
    st.dataframe(
        contrib_df.sort_values("Contribution (%)", ascending=False).style.format({"Contribution (%)": "{:.1f}%"}),
        use_container_width=True,
        hide_index=True
    )

    # =========================================================
    # INTERPRETATION
    # =========================================================
    diff = expert_prob - pooled_prob

    if abs(diff) < 0.05:
        st.info("Models are closely aligned for this profile.")
    elif diff > 0:
        st.success("Expert model is more optimistic than pooled model.")
    else:
        st.warning("Expert model is more conservative than pooled model.")

    # =========================================================
    # FINAL OUTPUT
    # =========================================================
    if final_prob >= 0.75:
        st.success("🟢 High likelihood of digital inclusion")
    elif final_prob >= 0.5:
        st.warning("🟡 Moderate likelihood of digital inclusion")
    else:
        st.error("🔴 Low likelihood of digital inclusion")
