import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import tempfile

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Digital Financial Inclusion Predictor", layout="wide")
st.title("🌍 Digital Financial Inclusion Predictor")
st.markdown("Analyze financial access using pooled or country-specific expert models.")

# ===============================
# LOAD MODELS FROM GITHUB
# ===============================
BASE_URL = "https://raw.githubusercontent.com/sitahlango-maker/Financial_Inclusion/main/"

def load_model(file_name):
    url = BASE_URL + file_name
    response = requests.get(url)

    if response.status_code != 200:
        st.error(f"Failed to load {file_name} from GitHub")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    return joblib.load(tmp_path)

@st.cache_resource
def load_all():
    model_pooled = load_model("model_pooled.pkl")
    gating_model = load_model("gating_model.pkl")
    experts = load_model("experts.pkl")
    feature_names = load_model("feature_names.pkl")
    return model_pooled, gating_model, experts, feature_names

model_pooled, gating_model, experts, feature_names = load_all()

# ===============================
# SIDEBAR SETTINGS
# ===============================
st.sidebar.header("Analysis Mode")

mode = st.sidebar.radio(
    "Select Model Type",
    ["Pooled Model Only", "Country Expert Model", "Compare Both"]
)

st.sidebar.header("Input Features")

# Build dynamic input
input_data = {}
for f in feature_names:
    input_data[f] = st.sidebar.number_input(f, value=0.0)

input_df = pd.DataFrame([input_data])
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# ===============================
# PREDICTION
# ===============================
if st.button("🔮 Run Analysis", use_container_width=True):

    with st.spinner("Running model..."):

        # --- pooled prediction ---
        pooled_prob = model_pooled.predict_proba(input_df)[0, 1]

        # --- gating model ---
        pred_country = gating_model.predict(input_df)[0]
        gating_conf = np.max(gating_model.predict_proba(input_df))

        expert_prob = None
        model_used = "Pooled Model"

        # --- expert routing ---
        if pred_country in experts:
            expert_model = experts[pred_country]
            expert_prob = expert_model.predict_proba(input_df)[0, 1]

        # ===============================
        # DISPLAY SECTION
        # ===============================
        st.subheader("📊 Results")

        def color(prob):
            if prob >= 0.75:
                return "green"
            elif prob >= 0.5:
                return "orange"
            else:
                return "red"

        # ===============================
        # MODE LOGIC
        # ===============================
        if mode == "Pooled Model Only":

            st.markdown(f"""
            <h2 style='color:{color(pooled_prob)}'>
            Probability: {pooled_prob:.1%}
            </h2>
            """, unsafe_allow_html=True)

            st.info("Using pooled model only")

        elif mode == "Country Expert Model":

            if expert_prob is None:
                st.warning("No expert available for this country. Using pooled model.")
                expert_prob = pooled_prob

            st.markdown(f"""
            <h2 style='color:{color(expert_prob)}'>
            Probability: {expert_prob:.1%}
            </h2>
            """, unsafe_allow_html=True)

            st.info(f"Expert selected: {pred_country} | Confidence: {gating_conf:.1%}")

        else:
            # Compare
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🧠 Pooled Model")
                st.markdown(f"<h2 style='color:{color(pooled_prob)}'>{pooled_prob:.1%}</h2>", unsafe_allow_html=True)

            with col2:
                st.markdown("### 🌍 Expert Model")

                if expert_prob is None:
                    st.warning("No expert available")
                else:
                    st.markdown(f"<h2 style='color:{color(expert_prob)}'>{expert_prob:.1%}</h2>", unsafe_allow_html=True)

        # ===============================
        # INTERPRETATION
        # ===============================
        st.markdown("### 📌 Interpretation")

        def explain(p):
            if p >= 0.75:
                return "🟢 High likelihood of financial access"
            elif p >= 0.5:
                return "🟠 Moderate likelihood"
            else:
                return "🔴 Low likelihood / barriers exist"

        st.write("Pooled:", explain(pooled_prob))

        if expert_prob is not None:
            st.write("Expert:", explain(expert_prob))
