# =========================================================
# routing.py
# =========================================================

import numpy as np
import pandas as pd
import joblib

# =========================================================
# LOAD MODELS
# =========================================================
model_pooled = joblib.load(
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

# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict_with_gating(
    X_input,
    return_routing_info=True
):

    # -----------------------------------------------------
    # ENSURE DATAFRAME
    # -----------------------------------------------------
    if not isinstance(X_input, pd.DataFrame):

        X_input = pd.DataFrame(
            X_input,
            columns=feature_names
        )

    # -----------------------------------------------------
    # ALIGN FEATURES
    # -----------------------------------------------------
    for col in feature_names:

        if col not in X_input.columns:

            X_input[col] = medians.get(col, 0)

    X_input = X_input.reindex(
        columns=feature_names,
        fill_value=0
    )

    # -----------------------------------------------------
    # ROUTING
    # -----------------------------------------------------
    try:

        predicted_country = gating_model.predict(
            X_input
        )

    except Exception:

        predicted_country = np.array(
            ["KEN"] * len(X_input)
        )

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------
    results = []

    routing_info = []

    # -----------------------------------------------------
    # LOOP THROUGH SAMPLES
    # -----------------------------------------------------
    for i, country in enumerate(predicted_country):

        sample = X_input.iloc[[i]]

        # -------------------------------------------------
        # POOLED PREDICTION
        # -------------------------------------------------
        pooled_prob = model_pooled.predict_proba(
            sample
        )[0,1]

        # -------------------------------------------------
        # DEFAULT
        # -------------------------------------------------
        expert_prob = pooled_prob

        model_used = "Pooled"

        # -------------------------------------------------
        # EXPERT
        # -------------------------------------------------
        if country in experts:

            try:

                expert_model = experts[country]

                # ALIGN TO EXPERT FEATURES
                if hasattr(
                    expert_model,
                    "feature_names_in_"
                ):

                    expert_features = list(
                        expert_model.feature_names_in_
                    )

                    sample_expert = sample.reindex(
                        columns=expert_features,
                        fill_value=0
                    )

                else:

                    sample_expert = sample

                # PREDICT
                expert_prob = expert_model.predict_proba(
                    sample_expert
                )[0,1]

                model_used = f"Expert_{country}"

            except Exception:

                expert_prob = pooled_prob

                model_used = (
                    f"Expert_{country}_Fallback"
                )

        # -------------------------------------------------
        # ENSEMBLE LOGIC
        # -------------------------------------------------
        final_prob = (
            pooled_prob * 0.7
            +
            expert_prob * 0.3
        )

        # -------------------------------------------------
        # SAVE RESULTS
        # -------------------------------------------------
        results.append(final_prob)

        routing_info.append(model_used)

    # =====================================================
    # RETURN
    # =====================================================
    if return_routing_info:

        return (
            np.array(results),
            routing_info
        )

    return np.array(results)
