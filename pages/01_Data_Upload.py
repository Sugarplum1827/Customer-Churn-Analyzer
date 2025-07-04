import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor
from utils.transaction_processor import TransactionProcessor

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
        
        # Check if this is transaction data that needs conversion
        transaction_columns = ['Name', 'Email', 'Product', 'Transaction Date']
        is_transaction_data = all(col in data.columns for col in transaction_columns)
        
        if is_transaction_data:
            st.info("📊 Detected transaction data. Converting to customer-level churn dataset...")
            
            # Allow user to configure conversion parameters
            col1, col2 = st.columns(2)
            with col1:
                churn_days = st.number_input(
                    "Days to define churn",
                    min_value=7,
                    max_value=365,
                    value=30,
                    help="Customers with no activity for this many days are considered churned"
                )
            with col2:
                # Auto-calculate observation date (30 days before max date by default)
                max_date = pd.to_datetime(data['Transaction Date']).max()
                default_obs_date = max_date - pd.Timedelta(days=churn_days)
                observation_date = st.date_input(
                    "Observation cutoff date",
                    value=default_obs_date.date(),
                    help="Date to split historical vs future data"
                )
            
            if st.button("🔄 Convert Transaction Data", type="primary"):
                with st.spinner("Converting transaction data to churn dataset..."):
                    processor = TransactionProcessor()
                    data = processor.convert_transactions_to_churn_data(
                        data, observation_date, churn_days
                    )
                    st.session_state.data = data
        else:
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
        st.write("Select the column that represents customer churn (must have exactly 2 unique values)")
        
        # Analyze columns to help user identify binary columns
        binary_columns = []
        binary_column_names = []
        for col in data.columns:
            unique_count = data[col].nunique()
            if unique_count == 2:
                values = data[col].unique()
                binary_columns.append(f"{col} (values: {', '.join(map(str, values))})")
                binary_column_names.append(col)
        
        if binary_columns:
            st.success(f"✅ Found {len(binary_columns)} binary columns suitable for churn prediction:")
            for col_info in binary_columns:
                st.write(f"• {col_info}")
            
            # Auto-select likely churn column
            default_target = None
            churn_keywords = ['churn', 'churned', 'left', 'exited', 'attrition']
            for col in binary_column_names:
                if any(keyword in col.lower() for keyword in churn_keywords):
                    default_target = col
                    break
            
            if default_target:
                st.info(f"💡 Automatically detected '{default_target}' as likely churn column")
                default_index = data.columns.tolist().index(default_target)
            else:
                default_index = 0
                
        else:
            st.warning("⚠️ No binary columns found. Churn prediction requires a column with exactly 2 unique values.")
            default_index = 0
        
        target_column = st.selectbox(
            "Select target column:",
            options=data.columns.tolist(),
            index=default_index if binary_columns else 0,
            help="Choose the column that indicates whether a customer churned (must have exactly 2 values)"
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
    st.write("Your CSV should contain customer features and a binary target column indicating churn. Example:")
    
    sample_data = pd.DataFrame({
        'customer_id': ['C001', 'C002', 'C003', 'C004'],
        'age': [25, 45, 35, 28],
        'tenure_months': [12, 36, 24, 8],
        'monthly_charges': [50.0, 80.0, 65.0, 45.0],
        'total_charges': [600.0, 2880.0, 1560.0, 360.0],
        'contract_type': ['Month-to-month', 'Two year', 'One year', 'Month-to-month'],
        'payment_method': ['Credit card', 'Bank transfer', 'Electronic check', 'Credit card'],
        'churn': [1, 0, 0, 1]  # Binary: 1 = churned, 0 = stayed
    })
    
    st.write("**Key requirements:**")
    st.write("• Target column must have exactly 2 unique values (e.g., 0/1, Yes/No, True/False)")
    st.write("• Include customer features like demographics, usage, and service information")
    st.write("• Remove any customer identifiers before analysis (they'll be ignored during preprocessing)")
    
    st.dataframe(sample_data, use_container_width=True)
    
    # Generate sample dataset button
    if st.button("📥 Download Sample Dataset", help="Download a sample CSV file to test the application"):
        # Create a larger sample dataset
        np.random.seed(42)
        n_samples = 1000
        
        sample_dataset = pd.DataFrame({
            'customer_id': [f'C{i:04d}' for i in range(1, n_samples + 1)],
            'age': np.random.randint(18, 70, n_samples),
            'tenure_months': np.random.randint(1, 72, n_samples),
            'monthly_charges': np.round(np.random.uniform(20, 120, n_samples), 2),
            'total_charges': np.round(np.random.uniform(50, 8000, n_samples), 2),
            'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
            'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'online_security': np.random.choice(['Yes', 'No'], n_samples),
            'tech_support': np.random.choice(['Yes', 'No'], n_samples),
            'streaming_tv': np.random.choice(['Yes', 'No'], n_samples),
            'paperless_billing': np.random.choice([0, 1], n_samples),
            'senior_citizen': np.random.choice([0, 1], n_samples)
        })
        
        # Create realistic churn based on features
        churn_prob = (
            0.1 +  # base rate
            0.3 * (sample_dataset['contract_type'] == 'Month-to-month') +
            0.2 * (sample_dataset['tenure_months'] < 12) +
            0.1 * (sample_dataset['monthly_charges'] > 80) +
            0.1 * (sample_dataset['payment_method'] == 'Electronic check') +
            0.1 * (sample_dataset['senior_citizen'] == 1)
        )
        sample_dataset['churn'] = np.random.binomial(1, np.minimum(churn_prob, 0.8), n_samples)
        
        csv = sample_dataset.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample Dataset (CSV)",
            data=csv,
            file_name="sample_churn_data.csv",
            mime="text/csv",
            help="This sample dataset contains 1000 customers with realistic churn patterns"
        )
