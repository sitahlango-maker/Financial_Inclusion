import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Digital Finance Access Predictor",
    page_icon="🌍💸",
    layout="wide"
)

# =========================================================
# CLEAN PROFESSIONAL UI
# =========================================================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #F7F9FC 0%,
        #EEF3F9 45%,
        #E6EDF7 100%
    );
}

.main .block-container {
    background: rgba(255,255,255,0.84);
    padding: 2.5rem;
    border-radius: 22px;
    box-shadow: 0 12px 35px rgba(0,0,0,0.08);
}

/* Inputs */
input, textarea, select {
    background-color: white !important;
    color: #1F2A44 !important;
}

.stSelectbox div,
.stNumberInput input,
.stTextInput input {
    background-color: white !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background-color: rgba(255,255,255,0.92);
    border-radius: 18px;
    padding: 1rem;
    border: 1px solid rgba(0,0,0,0.05);
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(
        90deg,
        #4F8BF9,
        #7A5CFA
    );
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 1.5rem;
    font-size: 16px;
    font-weight: 600;
}

.stButton>button:hover {
    opacity: 0.92;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================
st.title("🌍💸 Digital Finance Access Predictor")

st.markdown("""
### East Africa Digital Financial Inclusion Intelligence System

This system predicts the likelihood of digital financial inclusion
using:

- 🌐 Pooled Regional Model
- 🧠 Country Expert Models
- 🧭 Dynamic Gating Architecture
""")

# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_models():

    pooled_model = joblib.load(
        "model_pooled.joblib"
    )

    experts = joblib.load(
        "experts.joblib"
    )

    gating_model = joblib.load(
        "gating_model.joblib"
    )

    feature_names = joblib.load(
        "feature_names.joblib"
    )

    medians = joblib.load(
        "medians.joblib"
    )

    return {
        "pooled": pooled_model,
        "experts": experts,
        "gating": gating_model,
        "features": feature_names,
        "medians": medians
    }

# =========================================================
# LOAD EVERYTHING
# =========================================================
try:
    models = load_models()

except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# =========================================================
# FEATURE LISTS
# =========================================================
FEATURES = models["features"]

# Gating model features
gating_features = [
    "female",
    "age",
    "educ",
    "inc_q",
    "urbanicity",
    "anydigpayment",
    "internet_use",
    "wgt",
    "reg_index",
    "reg_cons_prot",
    "reg_kyc_prop",
    "reg_entry_lim",
    "reg_max_lim",
    "reg_agent_el",
    "num_providers",
    "earliest_launch"
]

# =========================================================
# COUNTRY DEFAULTS
# =========================================================
country_defaults = {

    "KEN": {
        "country_name": "Kenya",
        "reg_index": 95,
        "reg_cons_prot": 90,
        "reg_kyc_prop": 92,
        "reg_entry_lim": 85,
        "reg_max_lim": 88,
        "reg_agent_el": 96,
        "num_providers": 7,
        "earliest_launch": 2007
    },

    "TZA": {
        "country_name": "Tanzania",
        "reg_index": 82,
        "reg_cons_prot": 80,
        "reg_kyc_prop": 78,
        "reg_entry_lim": 75,
        "reg_max_lim": 77,
        "reg_agent_el": 84,
        "num_providers": 6,
        "earliest_launch": 2008
    },

    "UGA": {
        "country_name": "Uganda",
        "reg_index": 79,
        "reg_cons_prot": 76,
        "reg_kyc_prop": 74,
        "reg_entry_lim": 73,
        "reg_max_lim": 72,
        "reg_agent_el": 80,
        "num_providers": 5,
        "earliest_launch": 2009
    }
}

# =========================================================
# USER INPUT
# =========================================================
st.subheader("👤 User Profile")

col1, col2, col3 = st.columns(3)

with col1:

    age = st.slider(
        "Age",
        min_value=18,
        max_value=80,
        value=30
    )

    gender = st.radio(
        "Gender",
        ["Male", "Female"]
    )

    location = st.radio(
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
        [0, 1, 2, 3, 4],
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
        format_func=lambda x: country_defaults[x]["country_name"]
    )

# =========================================================
# COUNTRY DATA
# =========================================================
country_data = country_defaults[country]

# =========================================================
# BUILD INPUT ROW
# =========================================================
row = {

    # Core user features
    "female": 1 if gender == "Female" else 0,
    "age": age,
    "educ": education,
    "inc_q": income,
    "urbanicity": 1 if location == "Urban" else 0,
    "internet_use": 1 if internet == "Yes" else 0,

    # Model-required placeholders
    "dig_account": 0,
    "anydigpayment": 0,
    "wgt": 1.0,

    # Country-level contextual features
    "reg_index": country_data["reg_index"],
    "reg_cons_prot": country_data["reg_cons_prot"],
    "reg_kyc_prop": country_data["reg_kyc_prop"],
    "reg_entry_lim": country_data["reg_entry_lim"],
    "reg_max_lim": country_data["reg_max_lim"],
    "reg_agent_el": country_data["reg_agent_el"],
    "num_providers": country_data["num_providers"],
    "earliest_launch": country_data["earliest_launch"]
}

# =========================================================
# CREATE DATAFRAME
# =========================================================
df = pd.DataFrame([row])

# =========================================================
# ALIGN FEATURES TO TRAINING SCHEMA
# =========================================================
for col in FEATURES:

    if col not in df.columns:
        df[col] = models["medians"].get(col, 0)

# Ensure exact order
df = df.reindex(
    columns=FEATURES,
    fill_value=0
)

# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict_models(input_df):

    pooled_model = models["pooled"]
    experts = models["experts"]
    gating_model = models["gating"]

    # =====================================================
    # GATING INPUT
    # =====================================================
    gating_input = input_df.reindex(
        columns=gating_features,
        fill_value=0
    )

    predicted_country = gating_model.predict(
        gating_input
    )[0]

    # =====================================================
    # POOLED MODEL
    # =====================================================
    pooled_prob = pooled_model.predict_proba(
        input_df
    )[0,1]

    # =====================================================
    # EXPERT MODEL
    # =====================================================
    expert_prob = pooled_prob
    routed_model = "Pooled"

    if predicted_country in experts:

        expert_model = experts[predicted_country]

        try:

            # Align to expert feature schema
            if hasattr(expert_model, "feature_names_in_"):

                expert_features = list(
                    expert_model.feature_names_in_
                )

                expert_input = input_df.reindex(
                    columns=expert_features,
                    fill_value=0
                )

            else:
                expert_input = input_df.copy()

            expert_prob = expert_model.predict_proba(
                expert_input
            )[0,1]

            routed_model = f"Expert_{predicted_country}"

        except Exception as e:
            routed_model = "Pooled_Fallback"

    return {
        "pooled_prob": pooled_prob,
        "expert_prob": expert_prob,
        "predicted_country": predicted_country,
        "routed_model": routed_model
    }

# =========================================================
# RUN PREDICTION
# =========================================================
if st.button("🔮 Predict Digital Inclusion"):

    try:

        results = predict_models(df)

        pooled = results["pooled_prob"]
        expert = results["expert_prob"]

        # =================================================
        # METRICS
        # =================================================
        st.markdown("---")
        st.subheader("📊 Prediction Results")

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric(
                "🌐 Pooled Model",
                f"{pooled*100:.1f}%"
            )

        with m2:
            st.metric(
                "🧠 Expert Model",
                f"{expert*100:.1f}%"
            )

        with m3:
            st.metric(
                "🧭 Routed To",
                results["routed_model"]
            )

        # =================================================
        # FINAL PROBABILITY
        # =================================================
        final_prob = expert

        st.markdown("### 🎯 Final Inclusion Probability")

        st.progress(float(final_prob))

        st.markdown(
            f"""
            ## {final_prob*100:.1f}%
            likelihood of digital financial inclusion
            """
        )

        # =================================================
        # INTERPRETATION
        # =================================================
        st.markdown("### 🧭 Interpretation")

        if final_prob >= 0.75:

            st.success("""
            🟢 High likelihood of digital financial inclusion.

            This profile strongly aligns with characteristics
            associated with digital financial access in East Africa.
            """)

        elif final_prob >= 0.50:

            st.warning("""
            🟡 Moderate likelihood of digital financial inclusion.

            Some enabling factors are present, but barriers may still exist.
            """)

        else:

            st.error("""
            🔴 Low likelihood of digital financial inclusion.

            This profile may face structural or socioeconomic barriers
            to digital financial access.
            """)

        # =================================================
        # ROUTING INFO
        # =================================================
        st.info(
            f"""
            🧭 Gating Model routed prediction to:
            {results['predicted_country']}
            """
        )

    except Exception as e:

        st.error(f"❌ Prediction error: {e}")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.caption("""
Digital Finance Access Predictor •
Mixture-of-Experts Architecture •
East Africa Financial Inclusion Research
""")
