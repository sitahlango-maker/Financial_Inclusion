# 🌍 Digital Finance Access Predictor

## Overview

This project investigates whether digital financial inclusion can be predicted using historical socioeconomic and demographic data from East Africa. The study compares multiple machine learning approaches to determine whether specialized country-specific models outperform traditional pooled models.

The system predicts the probability that an individual has access to a digital financial account and provides explainable insights into the factors influencing the prediction.



## Research Objective

The study seeks to answer the following questions:

1. Can digital financial inclusion be predicted using historical data?
2. Does a country-specific Mixture of Experts (MoE) architecture outperform a traditional pooled model?
3. Does harmonizing country data improve predictive performance?
4. Which socioeconomic factors contribute most to digital financial inclusion?



## Models Evaluated

### 1. Pooled Model

A single XGBoost model trained on data from all countries combined.

### 2. Harmonized Model

An XGBoost model trained on a balanced dataset where each country contributes an equal number of observations.

### 3. Expert Models

Country-specific models trained separately for:

* Kenya (KEN)
* Tanzania (TZA)
* Uganda (UGA)

### 4. Routing Model

A Random Forest classifier that dynamically selects the most appropriate model for each observation.

### 5. Mixture of Experts (MoE)

The final architecture combining:

Input → Router → Best Model → Prediction



## Repository Structure

```text
Financial_Inclusion/

├── app.py
├── routing.py
├── requirements.txt
├── runtime.txt
├── README.md

├── feature_columns.joblib

├── pooled_model.joblib
├── harmonized_model.joblib

├── expert_model_KEN.joblib
├── expert_model_TZA.joblib
├── expert_model_UGA.joblib

├── routing_model.joblib

├── final_model_comparison.csv
├── feature_impact_table.csv

├── final_model_comparison_chart.png
├── feature_impact_comparison.png
├── shap_feature_impact_pooled.png

└── cleanfinancialinclusion.ipynb
```



## Technologies Used

* Python
* Streamlit
* XGBoost
* Scikit-Learn
* SHAP
* Pandas
* NumPy
* Plotly
* Matplotlib

---

## Running the Application

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app.py
```



## Application Features

* Predict digital financial inclusion probability
* Compare pooled, harmonized and expert models
* Dynamic routing through a Mixture of Experts architecture
* Feature importance visualisation
* Explainable AI using SHAP
* Executive dashboard for decision support



## Key Outputs

The platform provides:

* Probability of digital financial inclusion
* Model comparison results
* Routing decision
* Feature contribution analysis
* Explainable AI insights



## Author

Winnie MejaSitah Lang'o

MSc Research Project

Digital Financial Inclusion Prediction using a Mixture of Experts Architecture for East Africa

2026
