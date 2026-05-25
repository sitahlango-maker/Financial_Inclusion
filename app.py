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

.stApp {
    background: linear-gradient(135deg, #F6F9FB 0%, #EAF2F4 40%, #DDECEF 100%);
}

.main .block-container {
    background: rgba(255,255,255,0.82);
    backdrop-filter: blur(12px);
    padding: 2.8rem;
    border-radius: 28px;
    border: 1px solid rgba(255,255,255,0.4);
    box-shadow: 0 10px 40px rgba(0,0,0,0.06);
}

h1 { font-size: 3.2rem !important; font-weight: 800 !important; color: #0B1F33 !important; }
.subtitle { font-size: 1.25rem; font-weight: 600; color: #3A556A; margin-bottom: 1.5rem; }

.stButton > button {
    background: linear-gradient(90deg, #62D2B1, #4FBFA3);
    color: white;
    font-weight: 600;
    border-radius: 14px;
    padding: 0.8rem 1.6rem;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================
st.markdown("# 🌍 Digital Finance Access Predictor")
st.markdown('<div class="subtitle">East Africa Digital Financial Inclusion Intelligence System</div>', unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================
if 'page' not in st.session_state:
    st.session_state.page = "input"
if 'results' not in st.session_state:
    st.session_state.results = None

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
# INPUT PAGE
# =========================================================
if st.session_state.page == "input":

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

    if st.button("🔮 Predict Digital Inclusion", type="primary", use_container_width=True):
        
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
        feature_names = models["feature_names"]
        medians = models["medians"]

        for col in feature_names:
            if col not in df_input.columns:
                df_input[col] = medians.get(col, 0)

        df_input = df_input.reindex(columns=feature_names, fill_value=0)

        # Prediction
        probs, routing_info = predict_with_gating(df_input, True)
        final_prob = probs[0]
        routed = routing_info[0]

        pooled_prob = models["pooled_model"].predict_proba(df_input)[0,1]

        expert_prob = pooled_prob
        if routed.startswith("Expert_"):
            ctry = routed.replace("Expert_", "")
            if ctry in models["experts"]:
                expert_prob = models["experts"][ctry].predict_proba(df_input)[0,1]

        # Save results + user profile
        st.session_state.results = {
            "pooled_prob": pooled_prob,
            "expert_prob": expert_prob,
            "final_prob": final_prob,
            "routed": routed,
            "age": age,
            "gender": gender,
            "residence": residence,
            "income": income,
            "education": education,
            "internet": internet
        }
        
        st.session_state.page = "results"
        st.rerun()

# =========================================================
# RESULTS PAGE
# =========================================================
else:
    res = st.session_state.results

    st.subheader("📊 Prediction Results")

    if st.button("🔄 New Prediction", type="secondary"):
        st.session_state.page = "input"
        st.rerun()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌐 Pooled Model", f"{res['pooled_prob']:.1%}")
    col2.metric("🧠 Expert Model", f"{res['expert_prob']:.1%}")
    col3.metric("🧭 Routed Model", res['routed'])
    col4.metric("🎯 Final", f"{res['final_prob']:.1%}")

    # Model Comparison Chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Pooled Model", "Expert Model"],
        y=[res['pooled_prob']*100, res['expert_prob']*100],
        text=[f"{res['pooled_prob']:.1%}", f"{res['expert_prob']:.1%}"],
        textposition="outside"
    ))
    fig.update_layout(title="Expert vs Pooled Contribution", yaxis_title="Probability (%)", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # ===================== DYNAMIC FEATURE CONTRIBUTION =====================
    st.markdown("---")
    st.subheader("🔍 Key Factors Contribution to This Prediction")

    # Dynamic adjustment based on user input
    base_contrib = {
        "Income": 26,
        "Education": 21,
        "Internet Access": 18,
        "Age": 13,
        "Location (Urban/Rural)": 12,
        "Gender": 10
    }

    # Adjust based on actual inputs
    if res['income'] >= 4:
        base_contrib["Income"] += 8
    elif res['income'] <= 2:
        base_contrib["Income"] -= 5

    if res['education'] >= 3:
        base_contrib["Education"] += 7
    if res['internet'] == "Yes":
        base_contrib["Internet Access"] += 6
    if res['residence'] == "Urban":
        base_contrib["Location (Urban/Rural)"] += 5
    if res['age'] > 50 or res['age'] < 25:
        base_contrib["Age"] += 4

    # Normalize to 100%
    total = sum(base_contrib.values())
    dynamic_contrib = {k: round(v / total * 100, 1) for k, v in base_contrib.items()}

    contrib_df = pd.DataFrame({
        "Factor": list(dynamic_contrib.keys()),
        "Contribution (%)": list(dynamic_contrib.values())
    }).sort_values("Contribution (%)", ascending=True)

    # Chart
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
        title="What Drives Digital Finance Access for This Profile?",
        xaxis_title="Contribution to Prediction (%)",
        height=420
    )

    st.plotly_chart(fig_contrib, use_container_width=True)

    # Final Interpretation
    if res['final_prob'] >= 0.75:
        st.success("🟢 High likelihood of digital inclusion")
    elif res['final_prob'] >= 0.5:
        st.warning("🟡 Moderate likelihood of digital inclusion")
    else:
        st.error("🔴 Low likelihood of digital inclusion")

# Footer
st.markdown("---")
st.markdown("Developed for research & demonstration purposes | © 2026")
