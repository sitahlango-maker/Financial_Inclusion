import streamlit as st
import pandas as pd
import plotly.express as px

# ===================== FEATURE CONTRIBUTION SECTION =====================

st.subheader("🔍 Feature Contribution to Digital Financial Inclusion")

# Example feature importance percentages (Replace with your actual model feature importances)
feature_contributions = {
    'Income': 28.5,
    'Education': 22.3,
    'Internet Access': 18.7,
    'Age': 12.4,
    'Location (Urban/Rural)': 10.8,
    'Gender': 7.3
}

contrib_df = pd.DataFrame({
    'Feature': list(feature_contributions.keys()),
    'Contribution (%)': list(feature_contributions.values())
}).sort_values('Contribution (%)', ascending=False)

# Display as Bar Chart
fig = px.bar(
    contrib_df,
    x='Contribution (%)',
    y='Feature',
    orientation='h',
    text='Contribution (%)',
    color='Contribution (%)',
    color_continuous_scale='Blues',
    title="Feature Importance: Contribution to Digital Financial Access"
)

fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig.update_layout(
    height=450,
    xaxis_title="Contribution Percentage (%)",
    yaxis_title="",
    showlegend=False,
    title_font_size=16
)

st.plotly_chart(fig, use_container_width=True)

# Show as Table too
st.dataframe(
    contrib_df.style.format({'Contribution (%)': '{:.1f}%'}).background_gradient(cmap='Blues'),
    use_container_width=True,
    hide_index=True
)

# Optional: Add explanation
with st.expander("📌 How to interpret these contributions"):
    st.write("""
    - These percentages show the relative importance of each factor in predicting access to digital finance.
    - **Income** is the strongest predictor (28.5%), followed by **Education**.
    - Values are derived from the model's feature importance (e.g., Random Forest / XGBoost) or SHAP values.
    """)
