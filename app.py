import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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
# LIGHT SWIFT-STYLE UI
# =========================================================
st.markdown("""
<style>

/* =====================================================
BACKGROUND
===================================================== */

.stApp {
    background:
    linear-gradient(
        135deg,
        #EDF4F7 0%,
        #DDEBEC 35%,
        #CDE3E0 70%,
        #BFE8DF 100%
    );
}

/* =====================================================
MAIN CONTAINER
===================================================== */

.main .block-container {

    background: rgba(255,255,255,0.75);

    backdrop-filter: blur(10px);

    padding: 2.5rem;

    border-radius: 24px;

    border: 1px solid rgba(255,255,255,0.35);

    box-shadow:
    0 8px 32px rgba(0,0,0,0.08);
}

/* =====================================================
HEADINGS
===================================================== */

h1, h2, h3, h4 {

    color: #1F2A44 !important;

    font-weight: 700 !important;
}

/* =====================================================
TEXT
===================================================== */

p, div, label, span {

    color: #2D3748 !important;

    font-size: 15px !important;
}

/* =====================================================
INPUTS
===================================================== */

input,
textarea,
select {

    background-color: white !important;

    color: #1F2A44 !important;

    border-radius: 12px !important;

    border: 1px solid #D6E2E5 !important;
}

/* =====================================================
SELECT BOX
===================================================== */

.stSelectbox div {

    background-color: white !important;

    color: #1F2A44 !important;
}

/* =====================================================
METRIC CARDS
===================================================== */

[data-testid="metric-container"] {

    background: rgba(255,255,255,0.90);

    border-radius: 18px;

    padding: 1rem;

    border: 1px solid #D8E5E7;

    box-shadow:
    0 4px 12px rgba(0,0,0,0.05);
}

/* =====================================================
BUTTON
===================================================== */

.stButton > button {

    background:
    linear-gradient(
        90deg,
        #88D8C3,
        #70CDB7
    );

    color: #1F2A44;

    border: none;

    border-radius: 12px;

    padding: 0.8rem 1.6rem;

    font-weight: 700;

    font-size: 16px;
}

.stButton > button:hover {

    background:
    linear-gradient(
        90deg,
        #9BE5D3,
        #82D9C5
    );

    color: #1F2A44;
}

/* =====================================================
PROGRESS BAR
===================================================== */

.stProgress > div > div > div {

    background-color: #70CDB7;
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
# LOAD METADATA
# =========================================================
@st.cache_resource
def load_metadata():

    return {

        "feature_names": joblib.load(
            "feature_names.joblib"
        ),

        "medians": joblib.load(
            "medians.joblib"
        ),

        "experts": joblib.load(
            "experts.joblib"
        ),

        "pooled": joblib.load(
            "model_pooled.joblib"
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

    # TEMPORARY COMPATIBILITY FEATURES
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
        if final_prob >= 0.75:

            st.success(
                "🟢 High likelihood of digital financial inclusion"
            )

        elif final_prob >= 0.50:

            st.warning(
                "🟡 Moderate likelihood of digital financial inclusion"
            )

        else:

            st.error(
                "🔴 Low likelihood of digital financial inclusion"
            )

        # =================================================
        # FEATURE IMPORTANCE
        # =================================================
        st.markdown("---")

        st.subheader(
            "🔍 Top Features Influencing Prediction"
        )

        if routed_model.startswith("Expert"):

            country_key = routed_model.replace(
                "Expert_",
                ""
            )

            model = models["experts"][country_key]

        else:

            model = models["pooled"]

        if hasattr(model, "feature_importances_"):

            importance = pd.Series(
                model.feature_importances_,
                index=feature_names
            )

            importance = importance.sort_values(
                ascending=False
            ).head(10)

            fig, ax = plt.subplots(
                figsize=(10,6)
            )

            sns.barplot(
                x=importance.values,
                y=importance.index,
                ax=ax
            )

            ax.set_title(
                "Top 10 Important Features"
            )

            ax.set_xlabel(
                "Importance Score"
            )

            ax.set_ylabel(
                "Features"
            )

            st.pyplot(fig)

    except Exception as e:

        st.error(f"❌ Prediction error: {e}")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.caption(
    "Digital Finance Access Predictor • Mixture-of-Experts Architecture • East Africa"
)
