import streamlit as st
import pandas as pd
import numpy as np
import joblib
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

/* =====================================================
IMPORT FONT
===================================================== */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* =====================================================
GLOBAL FONT
===================================================== */

html, body, [class*="css"] {

    font-family: 'Inter', sans-serif;
}

/* =====================================================
BACKGROUND
===================================================== */

.stApp {

    background:
    linear-gradient(
        135deg,
        #F4F8FA 0%,
        #E7F0F2 35%,
        #D8ECE8 70%,
        #CDEEE6 100%
    );
}

/* =====================================================
MAIN CONTAINER
===================================================== */

.main .block-container {

    background: rgba(255,255,255,0.78);

    backdrop-filter: blur(14px);

    padding: 2.8rem;

    border-radius: 28px;

    border: 1px solid rgba(255,255,255,0.45);

    box-shadow:
    0 10px 40px rgba(0,0,0,0.07);
}

/* =====================================================
HEADINGS
===================================================== */

h1 {

    color: #102A43 !important;

    font-size: 2.5rem !important;

    font-weight: 700 !important;
}

h2, h3, h4 {

    color: #243B53 !important;

    font-weight: 600 !important;
}

/* =====================================================
TEXT
===================================================== */

p, div, span {

    color: #334E68 !important;

    font-size: 15px !important;
}

/* =====================================================
LABELS
===================================================== */

label {

    color: #243B53 !important;

    font-weight: 600 !important;
}

/* =====================================================
SELECT BOX
===================================================== */

.stSelectbox > div > div {

    background: rgba(255,255,255,0.95) !important;

    border-radius: 14px !important;

    border: 1px solid #D9E2EC !important;

    color: #102A43 !important;

    min-height: 48px;
}

/* =====================================================
INPUTS
===================================================== */

input {

    background: rgba(255,255,255,0.95) !important;

    border-radius: 14px !important;

    border: 1px solid #D9E2EC !important;

    color: #102A43 !important;
}

/* =====================================================
RADIO BUTTONS
===================================================== */

.stRadio {

    background: rgba(255,255,255,0.45);

    padding: 0.7rem;

    border-radius: 14px;
}

/* =====================================================
METRICS
===================================================== */

[data-testid="metric-container"] {

    background: rgba(255,255,255,0.90);

    border-radius: 20px;

    padding: 1.2rem;

    border: 1px solid #D9E2EC;

    box-shadow:
    0 4px 14px rgba(0,0,0,0.04);
}

/* =====================================================
BUTTON
===================================================== */

.stButton > button {

    background:
    linear-gradient(
        90deg,
        #6FD3C0,
        #56C6B0
    );

    color: white !important;

    border: none;

    border-radius: 16px;

    padding: 0.9rem 1.8rem;

    font-size: 16px;

    font-weight: 600;

    transition: all 0.25s ease;
}

.stButton > button:hover {

    transform: translateY(-1px);

    box-shadow:
    0 8px 20px rgba(86,198,176,0.25);
}

/* =====================================================
PROGRESS BAR
===================================================== */

.stProgress > div > div > div {

    background-color: #56C6B0;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================
st.title("🌍 Digital Finance Access Predictor")

st.markdown("""
### East Africa Digital Financial Inclusion Intelligence System

This platform predicts the likelihood of digital financial inclusion using:

- 🌐 Regional Pooled Intelligence
- 🧠 Country Expert Models
- 🧭 Dynamic Routing Architecture
""")

# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_metadata():

    return {

        "feature_names": joblib.load(
            "feature_names.joblib"
        ),

        "medians": joblib.load(
            "medians.joblib"
        )
    }

try:

    models = load_metadata()

except Exception as e:

    st.error(f"❌ Model loading failed: {e}")

    st.stop()

# =========================================================
# COUNTRY SETTINGS
# =========================================================
country_defaults = {

    "KEN": {

        "name": "Kenya",

        "reg_index": 95,

        "num_providers": 7,

        "earliest_launch": 2007
    },

    "TZA": {

        "name": "Tanzania",

        "reg_index": 82,

        "num_providers": 5,

        "earliest_launch": 2008
    },

    "UGA": {

        "name": "Uganda",

        "reg_index": 78,

        "num_providers": 4,

        "earliest_launch": 2009
    }
}

# =========================================================
# USER INPUT
# =========================================================
st.subheader("👤 Enter User Profile")

col1, col2, col3 = st.columns(3)

with col1:

    age = st.slider(
        "Age",
        18,
        80,
        30
    )

    gender = st.radio(
        "Gender",
        ["Male", "Female"]
    )

    residence = st.radio(
        "Residence",
        ["Rural", "Urban"]
    )

with col2:

    income = st.selectbox(
        "Income Quintile",
        [1, 2, 3, 4, 5]
    )

    education = st.selectbox(
        "Education Level",
        [0,1,2,3,4],
        format_func=lambda x: {
            0: "No Education",
            1: "Primary",
            2: "Secondary",
            3: "Tertiary",
            4: "Higher"
        }[x]
    )

    internet = st.radio(
        "Internet Use",
        ["No", "Yes"]
    )

with col3:

    country = st.selectbox(
        "Country",
        ["KEN", "TZA", "UGA"],
        format_func=lambda x:
        country_defaults[x]["name"]
    )

# =========================================================
# BUILD INPUT
# =========================================================
country_data = country_defaults[country]

row = {

    "female": 1 if gender == "Female" else 0,

    "age": age,

    "educ": education,

    "inc_q": income,

    "urbanicity": 1 if residence == "Urban" else 0,

    "internet_use": 1 if internet == "Yes" else 0,

    # TEMPORARY MODEL COMPATIBILITY
    "dig_account": 0,

    "anydigpayment": 0,

    "wgt": 1.0,

    # COUNTRY FEATURES
    "reg_index": country_data["reg_index"],

    "num_providers": country_data["num_providers"],

    "earliest_launch": country_data["earliest_launch"]
}

df_input = pd.DataFrame([row])

# =========================================================
# FEATURE ALIGNMENT
# =========================================================
feature_names = models["feature_names"]

medians = models["medians"]

for col in feature_names:

    if col not in df_input.columns:

        df_input[col] = medians.get(col, 0)

df_input = df_input.reindex(
    columns=feature_names,
    fill_value=0
)

# =========================================================
# RUN PREDICTION
# =========================================================
if st.button("🔮 Predict Digital Inclusion"):

    try:

        probs, routing_info = predict_with_gating(
            df_input,
            return_routing_info=True
        )

        final_prob = probs[0]

        routed_model = routing_info[0]

        # =================================================
        # RESULTS
        # =================================================
        st.markdown("---")

        st.subheader("📊 Prediction Results")

        m1, m2, m3 = st.columns(3)

        with m1:

            st.metric(
                "🌐 Final Probability",
                f"{final_prob:.1%}"
            )

        with m2:

            st.metric(
                "🧭 Routed To",
                routed_model
            )

        with m3:

            if final_prob >= 0.75:

                category = "High"

            elif final_prob >= 0.50:

                category = "Moderate"

            else:

                category = "Low"

            st.metric(
                "📌 Inclusion Level",
                category
            )

        # =================================================
        # PROGRESS BAR
        # =================================================
        st.progress(float(final_prob))

        # =================================================
        # INTERPRETATION
        # =================================================
        st.markdown("### 🧭 Personalized Inclusion Drivers")

        positive_factors = []

        risk_factors = []

        # POSITIVE FACTORS
        if education >= 3:

            positive_factors.append(
                "✅ Higher education level improves digital financial readiness"
            )

        if residence == "Urban":

            positive_factors.append(
                "✅ Urban residence improves access to digital infrastructure"
            )

        if income >= 3:

            positive_factors.append(
                "✅ Middle-to-high income improves financial accessibility"
            )

        if country == "KEN":

            positive_factors.append(
                "✅ Kenya's mature mobile money ecosystem supports inclusion"
            )

        if internet == "Yes":

            positive_factors.append(
                "✅ Internet access significantly supports digital finance usage"
            )

        # RISK FACTORS
        if internet == "No":

            risk_factors.append(
                "⚠ Lack of internet access may reduce digital service adoption"
            )

        if residence == "Rural":

            risk_factors.append(
                "⚠ Rural residence may reduce proximity to digital financial services"
            )

        if income <= 2:

            risk_factors.append(
                "⚠ Lower income levels may limit access to financial technologies"
            )

        if education <= 1:

            risk_factors.append(
                "⚠ Lower education levels may reduce digital financial literacy"
            )

        # DISPLAY POSITIVE FACTORS
        if positive_factors:

            st.success("\n\n".join(positive_factors))

        # DISPLAY RISK FACTORS
        if risk_factors:

            st.warning("\n\n".join(risk_factors))

        # FINAL SUMMARY
        st.markdown("---")

        if final_prob >= 0.75:

            st.success(
                "🟢 This profile demonstrates a strong likelihood of digital financial inclusion."
            )

        elif final_prob >= 0.50:

            st.warning(
                "🟡 This profile demonstrates moderate likelihood of digital financial inclusion."
            )

        else:

            st.error(
                "🔴 This profile may face structural or socioeconomic barriers to digital financial inclusion."
            )

    except Exception as e:

        st.error(f"❌ Prediction error: {e}")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.caption(
    "Digital Finance Access Predictor • Mixture-of-Experts Architecture • East Africa"
)
