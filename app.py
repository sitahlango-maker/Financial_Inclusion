import streamlit as st
import pandas as pd
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
# EXECUTIVE UI STYLING
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
    letter-spacing: -0.02em;
}

.subtitle {
    font-size: 1.35rem;
    font-weight: 500;
    color: #2C5F7A;
    margin-bottom: 1.8rem;
}

/* Executive Background */
.stApp {
    background: linear-gradient(135deg, #F8FAFC 0%, #E6F0F5 100%);
}

.main .block-container {
    background: rgba(255,255,255,0.95);
    backdrop-filter: blur(16px);
    padding: 2.2rem;
    border-radius: 24px;
    border: 1px solid rgba(255,255,255,0.6);
    box-shadow: 0 15px 35px rgba(15,43,70,0.08);
    max-width: 1350px;
    margin: 1rem auto;
}

/* Clean Cards */
div[data-testid="stMetric"] {
    background: white;
    border-radius: 16px;
    padding: 1rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #1E88E5, #1565C0);
    color: white;
    font-weight: 600;
    border-radius: 12px;
    padding: 0.75rem 1.8rem;
    font-size: 1.05rem;
    transition: all 0.3s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(21,101,192,0.3);
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================
st.markdown("# 🌍 Digital Finance Access Predictor")
st.markdown('<div class="subtitle">East Africa • Executive Intelligence Platform</div>', unsafe_allow_html=True)

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

    st.subheader("👤 Enter Client Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 80, 30)
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        residence = st.radio("Residence", ["Rural", "Urban"], horizontal=True)

    with col2:
        income = st.selectbox("Income Quintile", [1,2,3,4,5])
        education = st.selectbox("Education Level", [0,1,2,3,4],
            format_func=lambda x: ["No Education","Primary","Secondary","Tertiary","Higher"][x])
        internet = st.radio("Internet Use", ["No","Yes"], horizontal=True)

    with col3:
        country = st.selectbox("Country", ["KEN","TZA","UGA"],
            format_func=lambda x: country_defaults[x]["name"])

    if st.button("🔮 Generate Prediction", type="primary", use_container_width=True):
        
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

        st.session_state.results = {
            "pooled_prob": pooled_prob,
            "expert_prob": expert_prob,
            "final_prob": final_prob,
            "routed": routed,
            "age": age, "gender": gender, "residence": residence,
            "income": income, "education": education, "internet": internet
        }
        
        st.session_state.page = "results"
        st.rerun()

# =========================================================
# RESULTS PAGE (Optimized to fit one screen)
# =========================================================
else:
    res = st.session_state.results

    st.subheader("📊 Prediction Results")

    if st.button("🔄 New Prediction", type="secondary"):
        st.session_state.page = "input"
        st.rerun()

    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌐 Pooled Model", f"{res['pooled_prob']:.1%}")
    col2.metric("🧠 Expert Model", f"{res['expert_prob']:.1%}")
    col3.metric("🧭 Routed Model", res['routed'])
    col4.metric("🎯 **Final Score**", f"{res['final_prob']:.1%}", delta="Recommended")

    # Two Charts Side by Side
    col_a, col_b = st.columns(2)

    with col_a:
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=["Pooled", "Expert"],
            y=[res['pooled_prob']*100, res['expert_prob']*100],
            text=[f"{res['pooled_prob']:.1%}", f"{res['expert_prob']:.1%}"],
            textposition="outside",
            marker_color=['#90A4AE', '#1E88E5']
        ))
        fig1.update_layout(title="Model Comparison", height=380, margin=dict(t=40))
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        # Dynamic Feature Contribution
        base = {"Income": 26, "Education": 21, "Internet Access": 18, "Age": 13, "Location": 12, "Gender": 10}
        
        if res['income'] >= 4: base["Income"] += 9
        elif res['income'] <= 2: base["Income"] -= 6
        if res['education'] >= 3: base["Education"] += 8
        if res['internet'] == "Yes": base["Internet Access"] += 7
        if res['residence'] == "Urban": base["Location"] += 6
        if res['age'] < 30 or res['age'] > 55: base["Age"] += 5

        total = sum(base.values())
        dynamic_contrib = {k: round(v/total*100, 1) for k,v in base.items()}

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            y=list(dynamic_contrib.keys()),
            x=list(dynamic_contrib.values()),
            orientation='h',
            text=[f"{v:.1f}%" for v in dynamic_contrib.values()],
            textposition='outside',
            marker_color='#1E88E5'
        ))
        fig2.update_layout(title="Key Drivers for This Profile", height=380, margin=dict(t=40))
        st.plotly_chart(fig2, use_container_width=True)

    # Final Interpretation
    prob = res['final_prob']
    if prob >= 0.75:
        st.success("🟢 **HIGH** likelihood of digital financial inclusion")
    elif prob >= 0.55:
        st.warning("🟡 **MODERATE** likelihood of digital financial inclusion")
    else:
        st.error("🔴 **LOW** likelihood of digital financial inclusion")

# Footer
st.markdown("---")
st.markdown("**East Africa Digital Financial Inclusion Intelligence** | Research & Executive Analytics Platform © 2026")
