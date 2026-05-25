import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Digital Finance Access Predictor",
    page_icon="🌍",
    layout="wide"
)

# =========================================================
# SWIFT-STYLE UI
# =========================================================
st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background:
    linear-gradient(
        135deg,
        #2F3B3F 0%,
        #556B70 40%,
        #7FA8A3 75%,
        #AEE6DB 100%
    );
}

/* MAIN CONTAINER */
.main .block-container {

    background: rgba(255,255,255,0.10);

    backdrop-filter: blur(10px);

    padding: 2.5rem;

    border-radius: 22px;

    border: 1px solid rgba(255,255,255,0.10);

    box-shadow:
    0 8px 32px rgba(0,0,0,0.15);
}

/* TEXT */
h1, h2, h3, h4, h5, h6, p, div, label {

    color: white !important;
}

/* INPUTS */
input,
textarea,
select {

    background-color: rgba(255,255,255,0.96) !important;

    color: #1F2A44 !important;

    border-radius: 10px !important;
}

/* SELECT BOX */
.stSelectbox div {

    background-color: rgba(255,255,255,0.96) !important;

    color: #1F2A44 !important;
}

/* METRICS */
[data-testid="metric-container"] {

    background: rgba(255,255,255,0.12);

    border-radius: 18px;

    padding: 1rem;

    border: 1px solid rgba(255,255,255,0.08);
}

/* BUTTON */
.stButton > button {

    background:
    linear-gradient(
        90deg,
        #AEE6DB,
        #8DD7C8
    );

    color: #1F2A44;

    border: none;

    border-radius: 12px;

    padding: 0.8rem 1.5rem;

    font-weight: 700;

    font-size: 16px;
}

.stButton > button:hover {

    background:
    linear-gradient(
        90deg,
        #C5F5EC,
        #9FE6D8
    );

    color: #1F2A44;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================
st.title("🌍 Digital Finance Access Predictor")

st.markdown("""
### East Africa Digital Financial Inclusion Intelligence System

This platform predicts digital financial inclusion using:

- 🌐 Regional Pooled Intelligence
- 🧠 Country Expert Models
- 🧭 Dynamic Routing Architecture
""")

# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_models():

    models = {

        "pooled": joblib.load(
            "model_pooled.joblib"
        ),

        "experts": joblib.load(
            "experts.joblib"
        ),

        "gating": joblib.load(
            "gating_model.joblib"
        ),

        "feature_names": joblib.load(
            "feature_names.joblib"
        ),

        "medians": joblib.load(
            "medians.joblib"
        )
    }

    return models

# =========================================================
# LOAD
# =========================================================
try:

    models = load_models()

    st.success("✅ Models loaded successfully")

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

c1, c2, c3 = st.columns(3)

with c1:

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

    location = st.radio(
        "Residence",
        ["Rural", "Urban"]
    )

with c2:

    income = st.selectbox(
        "Income Quintile",
        [1,2,3,4,5]
    )

    education = st.selectbox(
        "Education Level",
        [0,1,2,3,4],
        format_func=lambda x: {
            0:"No Education",
            1:"Primary",
            2:"Secondary",
            3:"Tertiary",
            4:"Higher"
        }[x]
    )

    internet = st.radio(
        "Internet Use",
        ["No","Yes"]
    )

with c3:

    country = st.selectbox(
        "Country",
        ["KEN","TZA","UGA"],
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

    "urbanicity": 1 if location == "Urban" else 0,

    "internet_use": 1 if internet == "Yes" else 0,

    "reg_index": country_data["reg_index"],

    "num_providers": country_data["num_providers"],

    "earliest_launch": country_data["earliest_launch"]
}

df_input = pd.DataFrame([row])

# =========================================================
# FEATURE ALIGNMENT
# =========================================================
FEATURES = models["feature_names"]

for col in FEATURES:

    if col not in df_input.columns:

        df_input[col] = models["medians"].get(col, 0)

df_input = df_input.reindex(
    columns=FEATURES,
    fill_value=0
)

# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict_with_gating(df):

    pooled_model = models["pooled"]

    experts = models["experts"]

    gating_model = models["gating"]

    # ROUTING
    routed_country = gating_model.predict(df)[0]

    # POOLED
    pooled_prob = pooled_model.predict_proba(df)[0,1]

    # DEFAULT
    expert_prob = pooled_prob

    routed_model = "Pooled"

    # EXPERT
    if routed_country in experts:

        try:

            expert_model = experts[routed_country]

            expert_prob = expert_model.predict_proba(
                df
            )[0,1]

            routed_model = f"Expert_{routed_country}"

        except:
            pass

    # ENSEMBLE
    final_prob = (
        pooled_prob * 0.7
        +
        expert_prob * 0.3
    )

    return (
        final_prob,
        pooled_prob,
        expert_prob,
        routed_model
    )

# =========================================================
# RUN PREDICTION
# =========================================================
if st.button("🔮 Predict Digital Inclusion"):

    try:

        (
            final_prob,
            pooled_prob,
            expert_prob,
            routed_model
        ) = predict_with_gating(df_input)

        st.markdown("---")

        st.subheader("📊 Prediction Results")

        m1, m2, m3 = st.columns(3)

        with m1:

            st.metric(
                "🌐 Pooled Model",
                f"{pooled_prob:.1%}"
            )

        with m2:

            st.metric(
                "🧠 Expert Model",
                f"{expert_prob:.1%}"
            )

        with m3:

            st.metric(
                "🧭 Routed To",
                routed_model
            )

        st.progress(float(final_prob))

        st.subheader(
            f"🎯 Final Probability: {final_prob:.1%}"
        )

        # INTERPRETATION
        if final_prob >= 0.75:

            st.success(
                "🟢 High likelihood of digital financial inclusion"
            )

        elif final_prob >= 0.50:

            st.warning(
                "🟡 Moderate likelihood"
            )

        else:

            st.error(
                "🔴 Low likelihood of digital financial inclusion"
            )

        # =====================================================
        # FEATURE IMPORTANCE
        # =====================================================
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

            feature_names = FEATURES

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

            st.pyplot(fig)

    except Exception as e:

        st.error(f"❌ Prediction error: {e}")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.caption(
    "Digital Finance Access Predictor • East Africa"
)
