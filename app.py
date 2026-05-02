# app.py

import streamlit as st
import pandas as pd
import joblib
import requests
import tempfile

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Digital Finance Predictor", layout="wide")
st.title("💜 Digital Finance Access Predictor")

# -------------------------------
# BASE URL (GitHub RAW files)
# -------------------------------
BASE_URL = "https://raw.githubusercontent.com/sitahlango-maker/Financial_Inclusion/main/"

# -------------------------------
# MODEL LOADER (CACHED + ROBUST)
# -------------------------------
@st.cache_resource
def load_model(file_name):
    url = BASE_URL + file_name
    
    try:
        response = requests.get(url, timeout=30)
    except Exception as e:
        st.error(f"Connection error while loading {file_name}: {e}")
        st.stop()

    if response.status_code != 200:
        st.error(f"Failed to load {file_name} from GitHub:\n{url}")
        st.stop()

    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        return joblib.load(tmp_path)

    except Exception as e:
        st.error(f"Error loading {file_name}: {e}")
        st.stop()


# -------------------------------
# LOAD MODELS
# -------------------------------
model_pooled = load_model("model_pooled.pkl")
gating_model = load_model("gating_model.pkl")
experts = load_model("experts.pkl")
feature_names = load_model("feature_names.pkl")

# Validate feature names
if feature_names is None or len(feature_names) == 0:
    st.error("Feature names missing — check feature_names.pkl")
    st.stop()

st.success("✅ Models loaded successfully")

# -------------------------------
# USER INPUT UI
# -------------------------------
st.sidebar.header("Enter User Profile")

input_data = {}

for feature in feature_names:
    input_data[feature] = st.sidebar.number_input(feature, value=0.0)

input_df = pd.DataFrame([input_data])

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🔮 Predict", use_container_width=True):

    with st.spinner("Running Mixture of Experts..."):

        try:
            # Gating model
            pred_country = gating_model.predict(input_df)[0]
            gating_conf = gating_model.predict_proba(input_df).max(axis=1)[0]

            # Model selection
            if pred_country in experts and gating_conf >= 0.40:
                final_model = experts[pred_country]
                model_used = f"Expert ({pred_country})"
            else:
                final_model = model_pooled
                model_used = "Pooled Model"

            # Final prediction
            prob = final_model.predict_proba(input_df)[0, 1]

            # -------------------------------
            # DISPLAY RESULTS
            # -------------------------------
            st.subheader("🎯 Prediction Result")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Probability of Access", f"{prob:.2%}")

            with col2:
                st.metric("Model Used", model_used)
                st.metric("Gating Confidence", f"{gating_conf:.2%}")

            # Interpretation
            if prob >= 0.75:
                st.success("🟢 High Chance of Financial Access")
            elif prob >= 0.5:
                st.info("🔵 Moderate Chance of Financial Access")
            else:
                st.warning("🟠 Low Chance of Financial Access")

        except Exception as e:
            st.error(f"Prediction error: {e}")
