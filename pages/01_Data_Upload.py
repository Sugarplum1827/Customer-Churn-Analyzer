import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor

st.title("📁 Data Upload & Preprocessing")

# File upload
uploaded_file = st.file_uploader(
    "Upload your customer data (CSV format)",
    type=['csv'],
    help="Upload a CSV file containing customer data with features and churn labels"
)

if uploaded_file is not None:
    try:
        # Load data
        data = pd.read_csv(uploaded_file)
        st.session_state.data = data
        
        st.success(f"✅ Data uploaded successfully! Shape: {data.shape}")
        
        # Display basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", data.shape[0])
        with col2:
            st.metric("Total Columns", data.shape[1])
        with col3:
            st.metric("Missing Values", data.isnull().sum().sum())
        
        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(data.head(10), use_container_width=True)
        
        # Data info
        st.subheader("📊 Data Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Types:**")
            dtype_df = pd.DataFrame({
                'Column': data.dtypes.index,
                'Data Type': data.dtypes.astype(str).values
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        with col2:
            st.write("**Missing Values:**")
            missing_df = pd.DataFrame({
                'Column': data.columns,
                'Missing Count': data.isnull().sum().values,
                'Missing %': (data.isnull().sum() / len(data) * 100).round(2).values
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0]
            if len(missing_df) > 0:
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.info("No missing values found!")
        
        # Target column selection
        st.subheader("🎯 Target Column Selection")
        st.write("Select the column that represents customer churn (0/1 or False/True)")
        
        target_column = st.selectbox(
            "Select target column:",
            options=data.columns.tolist(),
            help="Choose the column that indicates whether a customer churned"
        )
        
        if target_column:
            st.session_state.target_column = target_column
            
            # Show target distribution
            target_dist = data[target_column].value_counts()
            st.write(f"**Target Distribution for '{target_column}':**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(target_dist.to_frame('Count'), use_container_width=True)
            with col2:
                st.bar_chart(target_dist)
        
        # Data preprocessing
        st.subheader("🔧 Data Preprocessing")
        
        if st.button("Process Data", type="primary"):
            with st.spinner("Processing data..."):
                processor = DataProcessor()
                processed_data, feature_columns = processor.preprocess_data(data, target_column)
                
                st.session_state.processed_data = processed_data
                st.session_state.feature_columns = feature_columns
                
                st.success("✅ Data preprocessing completed!")
                
                # Show processed data info
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Processed Features", len(feature_columns))
                with col2:
                    st.metric("Final Dataset Size", processed_data.shape[0])
                
                st.write("**Feature Columns:**")
                st.write(feature_columns)
                
                # Show processed data preview
                st.write("**Processed Data Preview:**")
                st.dataframe(processed_data.head(), use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure your CSV file is properly formatted")

else:
    st.info("👆 Please upload a CSV file to get started")
    
    # Sample data format
    st.subheader("📝 Expected Data Format")
    st.write("Your CSV should contain customer features and a target column indicating churn. Example:")
    
    sample_data = pd.DataFrame({
        'customer_id': ['C001', 'C002', 'C003'],
        'age': [25, 45, 35],
        'tenure': [12, 36, 24],
        'monthly_charges': [50.0, 80.0, 65.0],
        'total_charges': [600.0, 2880.0, 1560.0],
        'contract_type': ['Month-to-month', 'Two year', 'One year'],
        'churn': [1, 0, 0]
    })
    
    st.dataframe(sample_data, use_container_width=True)
