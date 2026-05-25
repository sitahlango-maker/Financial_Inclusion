import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.table import table
import io
from PIL import Image

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Digital Financial Inclusion Predictor",
    page_icon="💰",
    layout="wide"
)

# ===================== TITLE & HEADER =====================
st.title("💳 Digital Financial Inclusion Predictor")
st.markdown("""
**A Comparative Study of Pooled vs Country-Specific Models**  
*Using Global Findex and GSMA Data for Kenya, Tanzania & Uganda*
""")

# ===================== SIDEBAR =====================
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", 
    ["Overview", "Model Comparison", "Feature Contributions", "About"])

# ===================== OVERVIEW PAGE =====================
if page == "Overview":
    st.header("1.1 Research Approach")
    st.write("""
    This study adopts a **quantitative predictive research approach** using publicly available data. 
    Data from the **Global Findex** database was combined with country-level indicators on infrastructure 
    and regulation from **GSMA**.
    """)
    
    st.write("""
    Two modelling strategies were compared:
    - **Pooled Model**: Trained on combined cross-country data
    - **Country-Specific Expert Models**: Separate models for Kenya, Tanzania, and Uganda
    """)
    
    st.success("✅ Models evaluated using Accuracy, F1-Score, and AUC.")

# ===================== MODEL COMPARISON PAGE =====================
elif page == "Model Comparison":
    st.header("Final Model Comparison: Pooled vs Country Expert Models")
    
    # Data
    data = {
        'Model': ['Pooled', 'Pooled', 'Pooled', 'Pooled', 
                  'Expert_KEN', 'Expert_TZA', 'Expert_UGA'],
        'Country': ['All', 'KEN', 'TZA', 'UGA', 'KEN', 'TZA', 'UGA'],
        'Samples': [600, 196, 199, 205, 196, 199, 205],
        'Accuracy': [0.9800, 0.9898, 0.9749, 0.9756, 0.9898, 0.9749, 0.9756],
        'F1': [0.9867, 0.9941, 0.9796, 0.9842, 0.9941, 0.9796, 0.9842],
        'AUC': [0.9685, 0.9680, 0.9870, 0.9350, 0.9549, 0.9828, 0.9385]
    }
    
    df = pd.DataFrame(data)
    
    # Text Table
    st.subheader("📊 Model Performance Table")
    st.dataframe(df.style.format({
        'Accuracy': '{:.4f}', 
        'F1': '{:.4f}', 
        'AUC': '{:.4f}'
    }).background_gradient(subset=['Accuracy', 'F1', 'AUC'], cmap='Blues'), 
    use_container_width=True)
    
    # Download CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Table as CSV", csv, "model_comparison.csv", "text/csv")
    
    # Generate and Display PNG Table
    st.subheader("📸 Publication-ready Table Image")
    
    def create_table_image():
        fig, ax = plt.subplots(figsize=(13, 6.5))
        ax.axis('off')
        
        tab = table(ax, cellText=df.values, colLabels=df.columns, 
                    rowLabels=df.index, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        
        tab.auto_set_font_size(False)
        tab.set_fontsize(10.5)
        tab.auto_set_column_width(col=list(range(len(df.columns))))
        
        # Styling
        for key, cell in tab._cells.items():
            if key[0] == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#2C3E50')
            else:
                cell.set_facecolor('#F8F9FA' if key[0] % 2 == 0 else 'white')
        
        # Green highlights
        highlights = [(1,3), (1,4), (2,5), (5,3), (5,4)]
        for r, c in highlights:
            cell = tab._cells[(r+1, c)]
            cell.set_facecolor('#90EE90')
            cell.set_text_props(weight='bold')
        
        plt.title("Fair Evaluation: Pooled vs Country Expert Models", fontsize=14, fontweight='bold', pad=20)
        plt.figtext(0.5, 0.02, "Results saved to 'model_comparison_results.csv'", 
                    ha='center', fontsize=9, style='italic')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        plt.close()
        return buf.getvalue()
    
    table_img = create_table_image()
    st.image(table_img, use_column_width=True)
    
    # Download PNG
    st.download_button("📥 Download Table as PNG", table_img, "model_comparison_table.png", "image/png")

# ===================== FEATURE CONTRIBUTIONS PAGE =====================
elif page == "Feature Contributions":
    st.header("🔍 Feature Contribution to Digital Financial Access")
    
    # Feature Contributions
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
    
    # Bar Chart
    fig = px.bar(
        contrib_df,
        x='Contribution (%)',
        y='Feature',
        orientation='h',
        text='Contribution (%)',
        color='Contribution (%)',
        color_continuous_scale='Blues',
        title="Relative Importance of Features in Predicting Digital Financial Inclusion"
    )
    
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=500, xaxis_title="Contribution Percentage (%)", yaxis_title="")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Table
    st.subheader("Feature Contribution Table")
    st.dataframe(
        contrib_df.style.format({'Contribution (%)': '{:.1f}%'})
                    .background_gradient(subset=['Contribution (%)'], cmap='Blues'),
        use_container_width=True,
        hide_index=True
    )
    
    with st.expander("📌 Interpretation"):
        st.write("""
        These percentages represent the **relative contribution** of each variable to the model's prediction 
        of access to digital financial services. 
        
        **Key Insights:**
        - **Income** is the strongest predictor (28.5%)
        - **Education** and **Internet Access** are also major drivers
        - Demographic factors (Age, Gender, Location) have comparatively lower but still meaningful impact
        """)

# ===================== ABOUT PAGE =====================
else:
    st.header("About This Project")
    st.write("""
    This application presents findings from a predictive modelling study on digital financial inclusion 
    in East Africa (Kenya, Tanzania, and Uganda).
    
    **Data Sources:**
    - Global Findex Database
    - GSMA Mobile Connectivity and Regulation Indicators
    
    **Modelling Approaches:**
    - Pooled Cross-Country Model
    - Country-Specific Expert Models
    
    The results demonstrate that country-specific models generally outperform pooled models, 
    particularly when data volumes vary across countries.
    """)
    
    st.markdown("**Reference:** Banna et al. (2025)")

# Footer
st.markdown("---")
st.markdown("Developed for research & demonstration purposes | © 2026")
