import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(page_title="Customer Churn Analysis Tool",
                   page_icon="📊",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Main page
st.title("📊 Customer Churn Analysis Tool")
st.markdown("""
Welcome to the Customer Churn Analysis Tool! This application helps you analyze customer data, 
build machine learning models to predict churn, and gain insights into customer behavior.

## Features:
- **Data Upload**: Upload and preview your customer data (CSV format)
- **Exploratory Analysis**: Visualize data patterns and relationships
- **Model Training**: Train and compare multiple ML models
- **Predictions**: Make churn predictions on new data

## Getting Started:
1. Navigate to the **Data Upload** page to upload your customer dataset
2. Explore your data in the **Exploratory Analysis** section
3. Train models in the **Model Training** section
4. Make predictions in the **Predictions** section

Use the sidebar to navigate between different sections of the application.
""")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if "models" not in st.session_state:
    st.session_state.models = {}
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = []
if 'target_column' not in st.session_state:
    st.session_state.target_column = None

# Sidebar status
st.sidebar.markdown("## Application Status")
if st.session_state.data is not None:
    st.sidebar.success(f"✅ Data loaded: {st.session_state.data.shape[0]} rows")
else:
    st.sidebar.warning("⚠️ No data loaded")

if st.session_state.models:
    st.sidebar.success(f"✅ Models trained: {len(st.session_state.models)}")
else:
    st.sidebar.info("ℹ️ No models trained yet")

st.sidebar.markdown("---")
st.sidebar.markdown("### Instructions")
st.sidebar.markdown("""
1. Upload your CSV file in **Data Upload**
2. Explore patterns in **Exploratory Analysis** 
3. Train models in **Model Training**
4. Make predictions in **Predictions**
""")
