# routing.py
import numpy as np
import pandas as pd
import joblib

# Load models once when module is imported
model_pooled = joblib.load('trained_models/model_pooled.joblib')
experts = joblib.load('trained_models/experts.joblib')
gating_model = joblib.load('trained_models/gating_model.joblib')
feature_names = joblib.load('trained_models/feature_names.joblib')

def predict_with_gating(X_input, return_routing_info=True):
    """
    Route input to the best model using the gating model.
    """
    if not isinstance(X_input, pd.DataFrame):
        X_input = pd.DataFrame(X_input, columns=feature_names)
    
    # Align features
    X_input = X_input.reindex(columns=feature_names, fill_value=0)
    
    # Get routing decision
    predicted_country = gating_model.predict(X_input)
    
    results = []
    routing_info = []
    
    for i, country in enumerate(predicted_country):
        sample = X_input.iloc[[i]]
        
        if country in experts:
            try:
                expert = experts[country]
                if hasattr(expert, 'feature_names_in_'):
                    sample = sample.reindex(columns=expert.feature_names_in_, fill_value=0)
                prob = expert.predict_proba(sample)[0, 1]
                model_used = f"Expert_{country}"
            except Exception:
                prob = model_pooled.predict_proba(sample)[0, 1]
                model_used = f"Expert_{country} (fallback)"
        else:
            prob = model_pooled.predict_proba(sample)[0, 1]
            model_used = "Pooled"
        
        results.append(prob)
        routing_info.append(model_used)
    
    if return_routing_info:
        return np.array(results), routing_info
    return np.array(results)
