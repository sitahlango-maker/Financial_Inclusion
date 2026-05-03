import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Digital Finance Access Predictor",
    page_icon="💜",
    layout="centered"
)

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    try:
        expert_models = {
            "Kenya": joblib.load("model_kenya.pkl"),
            # Add more when available
            # "Nigeria": joblib.load("model_nigeria.pkl"),
        }
        pooled_model = joblib.load("model_pooled.pkl")
        return expert_models, pooled_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

expert_models, pooled_model = load_models()

# -----------------------------
# TITLE
# -----------------------------
st.title("💜 Digital Finance Access Predictor")

if expert_models is None:
    st.stop()

st.success("✅ Models loaded successfully")

# -----------------------------
# USER INPUT SECTION
# -----------------------------
st.header("🧾 Enter User Profile")

# Gender
gender = st.selectbox("Gender", ["Male", "Female"])
female = 1 if gender == "Female" else 0

# Age
age = st.slider("Age", 18, 80, 30)

# Education
education = st.selectbox(
    "Highest Education Level",
    ["No formal education", "Primary", "Secondary", "Tertiary"]
)

educ_map = {
    "No formal education": 0,
    "Primary": 1,
    "Secondary": 2,
    "Tertiary": 3
}
educ = educ_map[education]

# Income
income = st.selectbox(
    "Income Level",
    ["Lowest (Q1)", "Low (Q2)", "Middle (Q3)", "High (Q4)", "Highest (Q5)"]
)

inc_map = {
    "Lowest (Q1)": 1,
    "Low (Q2)": 2,
    "Middle (Q3)": 3,
    "High (Q4)": 4,
    "Highest (Q5)": 5
}
inc_q = inc_map[income]

# Urban / Rural
location = st.selectbox("Location", ["Rural", "Urban"])
urban = 1 if location == "Urban" else 0

# Digital account
dig_account_ui = st.selectbox("Do you have a digital account?", ["No", "Yes"])
dig_account = 1 if dig_account_ui == "Yes" else 0

# Digital payments
dig_payment_ui = st.selectbox("Have you made a digital payment?", ["No", "Yes"])
anydigpayment = 1 if dig_payment_ui == "Yes" else 0

# -----------------------------
# MODEL SELECTION
# -----------------------------
st.header("🌍 Model Selection")

country = st.selectbox("Select Country", list(expert_models.keys()))
model_type = st.radio("Model Type", ["Expert", "Pooled"])

# -----------------------------
# PREPARE INPUT
# -----------------------------
input_data = pd.DataFrame([{
    "female": female,
    "age": age,
    "educ": educ,
    "inc_q": inc_q,
    "urban": urban,
    "dig_account": dig_account,
    "anydigpayment": anydigpayment
}])

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔮 Predict Access"):

    # Select model
    if model_type == "Expert":
        model = expert_models[country]
        model_used = f"Expert ({country})"
    else:
        model = pooled_model
        model_used = "Pooled Model"

    try:
        prob = model.predict_proba(input_data)[0][1]
        prob_percent = prob * 100

        # -----------------------------
        # OUTPUT
        # -----------------------------
        st.subheader("🎯 Prediction Result")

        st.metric("Probability of Access", f"{prob_percent:.2f}%")
        st.write(f"**Model Used:** {model_used}")

        # Risk classification
        if prob_percent < 30:
            st.error("🔴 Very Low Access Likelihood")
        elif prob_percent < 60:
            st.warning("🟠 Moderate Access Gap")
        else:
            st.success("🟢 High Access Likelihood")

    except Exception as e:
        st.error(f"Prediction error: {e}")
