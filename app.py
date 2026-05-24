import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import tempfile
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="Digital Financial Inclusion Predictor", layout="wide")
st.title("🌍 Digital Financial Inclusion Predictor")
st.markdown("Mixture of Experts vs Pooled Model (Aligned Version)")

# ===============================
# LOAD MODELS
# ===============================
BASE_URL = "https://raw.githubusercontent.com/sitahlango-maker/Financial_Inclusion/main/"

def load_model(file):
    url = BASE_URL + file
    r = requests.get(url)

    if r.status_code != 200:
        st.error(f"Failed to load {file}")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(r.content)
        return joblib.load(tmp.name)

@st.cache_resource
def load_all():
    return (
        load_model("model_pooled.pkl"),
        load_model("gating_model.pkl"),
        load_model("experts.pkl")
    )

model_pooled, gating_model, experts = load_all()

st.success("Models loaded successfully ✅")

# ===============================
# FIXED FEATURE SET (CRITICAL)
# ===============================
feature_names = [
    'female', 'age', 'educ', 'inc_q', 'urbanicity',
    'dig_account', 'anydigpayment', 'internet_use',
    'wgt', 'reg_index', 'reg_cons_prot', 'reg_kyc_prop',
    'reg_entry_lim', 'reg_max_lim', 'reg_agent_el',
    'num_providers', 'earliest_launch'
]

# ===============================
# INPUT UI
# ===============================
st.subheader("👤 Enter User Profile")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 80, 30)
    gender = st.radio("Gender", ["Male", "Female"])
    urban = st.radio("Location", ["Rural", "Urban"])

with col2:
    inc_q = st.selectbox("Income Quintile", [1,2,3,4,5])
    educ = st.selectbox("Education Level", [0,1,2,3,4],
        format_func=lambda x: ["No Education","Primary","Secondary","Tertiary","Higher"][x])
    internet = st.radio("Uses Internet", ["No","Yes"])

with col3:
    country = st.selectbox("Country", ["KEN","TZA","UGA"])

# ===============================
# INPUT BUILDING (MATCH TRAINING)
# ===============================
input_dict = {
    "female": 1 if gender == "Female" else 0,
    "age": age,
    "educ": educ,
    "inc_q": inc_q,
    "urbanicity": 1 if urban == "Urban" else 0,
    "dig_account": 0,
    "anydigpayment": 0,
    "internet_use": 1 if internet == "Yes" else 0,
    "wgt": 1,
    "reg_index": 0,
    "reg_cons_prot": 0,
    "reg_kyc_prop": 0,
    "reg_entry_lim": 0,
    "reg_max_lim": 0,
    "reg_agent_el": 0,
    "num_providers": 0,
    "earliest_launch": 0
}

input_df = pd.DataFrame([input_dict])

# enforce correct order
input_df = input_df[feature_names]

# ===============================
# HELPERS
# ===============================
def color(p):
    if p > 0.75:
        return "#16a34a"
    elif p > 0.5:
        return "#f59e0b"
    return "#dc2626"

def box(title, value, c):
    st.markdown(f"""
    <div style="padding:15px;border-radius:10px;border:1px solid #ddd;text-align:center">
    <h4>{title}</h4>
    <h1 style="color:{c}">{value:.1%}</h1>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# FEATURE IMPORTANCE
# ===============================
def plot_importance(model, title):
    imp = model.feature_importances_

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": imp
    }).sort_values("importance", ascending=True).tail(10)

    fig, ax = plt.subplots()
    ax.barh(df["feature"], df["importance"])
    ax.set_title(title)
    st.pyplot(fig)

# ===============================
# PREDICTION
# ===============================
if st.button("🔮 Predict"):

    # gating
    pred_country = gating_model.predict(input_df)[0]
    conf = np.max(gating_model.predict_proba(input_df))

    # expert or pooled
    if pred_country in experts and conf > 0.4:
        model = experts[pred_country]
        label = f"Expert ({pred_country})"
    else:
        model = model_pooled
        label = "Pooled Model"

    prob = model.predict_proba(input_df)[0,1]
    pooled_prob = model_pooled.predict_proba(input_df)[0,1]

    st.markdown("## Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        box("Selected Model", prob, color(prob))
        st.caption(label)

    with col2:
        box("Pooled Model", pooled_prob, color(pooled_prob))

    with col3:
        st.metric("Gating Confidence", f"{conf:.1%}")
        st.metric("Predicted Country", pred_country)

    st.markdown("---")
    st.subheader("📊 Feature Importance")

    plot_importance(model_pooled, "Pooled Model Importance")

    if pred_country in experts:
        plot_importance(experts[pred_country], f"{pred_country} Expert Importance")
