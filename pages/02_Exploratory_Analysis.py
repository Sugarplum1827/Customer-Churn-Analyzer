import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from utils.visualizations import ChurnVisualizer

st.title("🔍 Exploratory Data Analysis")

if st.session_state.data is None:
    st.warning("⚠️ Please upload data first in the Data Upload section")
    st.stop()

data = st.session_state.data
target_column = st.session_state.target_column

if target_column is None:
    st.warning("⚠️ Please select a target column in the Data Upload section")
    st.stop()

visualizer = ChurnVisualizer()

# Overview metrics
st.subheader("📊 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Customers", len(data))
with col2:
    churn_count = data[target_column].sum()
    st.metric("Churned Customers", churn_count)
with col3:
    churn_rate = (churn_count / len(data) * 100)
    st.metric("Churn Rate", f"{churn_rate:.1f}%")
with col4:
    st.metric("Features", len(data.columns) - 1)

# Target distribution
st.subheader("🎯 Churn Distribution")
fig_churn = visualizer.plot_churn_distribution(data, target_column)
st.plotly_chart(fig_churn, use_container_width=True)

# Numerical features analysis
st.subheader("📈 Numerical Features Analysis")

numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
if target_column in numeric_cols:
    numeric_cols.remove(target_column)

if numeric_cols:
    selected_numeric = st.multiselect(
        "Select numerical features to analyze:",
        numeric_cols,
        default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
    )
    
    if selected_numeric:
        # Box plots for numerical features
        fig_box = visualizer.plot_numerical_features_boxplot(data, selected_numeric, target_column)
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("🔥 Correlation Heatmap")
        correlation_cols = selected_numeric + [target_column]
        fig_corr = visualizer.plot_correlation_heatmap(data[correlation_cols])
        st.plotly_chart(fig_corr, use_container_width=True)

# Categorical features analysis
st.subheader("📊 Categorical Features Analysis")

categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

if categorical_cols:
    selected_categorical = st.multiselect(
        "Select categorical features to analyze:",
        categorical_cols,
        default=categorical_cols[:4] if len(categorical_cols) >= 4 else categorical_cols
    )
    
    if selected_categorical:
        # Churn rate by categorical features
        fig_cat = visualizer.plot_categorical_churn_analysis(data, selected_categorical, target_column)
        st.plotly_chart(fig_cat, use_container_width=True)

# Advanced analysis
st.subheader("🔬 Advanced Analysis")

analysis_type = st.selectbox(
    "Select analysis type:",
    ["Feature Distributions", "Churn Rate by Segments", "Statistical Summary"]
)

if analysis_type == "Feature Distributions":
    if numeric_cols:
        feature_to_plot = st.selectbox("Select feature for distribution analysis:", numeric_cols)
        fig_dist = visualizer.plot_feature_distribution(data, feature_to_plot, target_column)
        st.plotly_chart(fig_dist, use_container_width=True)

elif analysis_type == "Churn Rate by Segments":
    if categorical_cols and len(categorical_cols) >= 2:
        col1_seg = st.selectbox("Select first categorical feature:", categorical_cols)
        col2_seg = st.selectbox("Select second categorical feature:", 
                               [col for col in categorical_cols if col != col1_seg])
        
        # Create segments and calculate churn rates
        segment_data = data.groupby([col1_seg, col2_seg])[target_column].agg(['count', 'sum']).reset_index()
        segment_data['churn_rate'] = (segment_data['sum'] / segment_data['count'] * 100).round(2)
        segment_data['segment'] = segment_data[col1_seg].astype(str) + " - " + segment_data[col2_seg].astype(str)
        
        fig_segments = px.bar(
            segment_data, 
            x='segment', 
            y='churn_rate',
            title=f'Churn Rate by {col1_seg} and {col2_seg}',
            labels={'churn_rate': 'Churn Rate (%)', 'segment': 'Segment'}
        )
        fig_segments.update_xaxis(tickangle=45)
        st.plotly_chart(fig_segments, use_container_width=True)
        
        # Show detailed table
        st.write("**Detailed Segment Analysis:**")
        display_df = segment_data[['segment', 'count', 'sum', 'churn_rate']].copy()
        display_df.columns = ['Segment', 'Total Customers', 'Churned', 'Churn Rate (%)']
        st.dataframe(display_df, use_container_width=True)

elif analysis_type == "Statistical Summary":
    st.write("**Statistical Summary by Churn Status:**")
    
    if numeric_cols:
        summary_stats = data.groupby(target_column)[numeric_cols].agg(['mean', 'median', 'std']).round(2)
        
        for col in numeric_cols[:3]:  # Show first 3 columns to avoid clutter
            st.write(f"**{col}:**")
            col_stats = summary_stats[col].T
            st.dataframe(col_stats, use_container_width=True)

# Data quality insights
st.subheader("🔍 Data Quality Insights")

col1, col2 = st.columns(2)

with col1:
    st.write("**Missing Values by Column:**")
    missing_info = data.isnull().sum().sort_values(ascending=False)
    missing_info = missing_info[missing_info > 0]
    
    if len(missing_info) > 0:
        fig_missing = px.bar(
            x=missing_info.values,
            y=missing_info.index,
            orientation='h',
            title='Missing Values Count',
            labels={'x': 'Missing Count', 'y': 'Columns'}
        )
        st.plotly_chart(fig_missing, use_container_width=True)
    else:
        st.success("No missing values found!")

with col2:
    st.write("**Data Types Distribution:**")
    dtype_counts = data.dtypes.value_counts()
    fig_dtypes = px.pie(
        values=dtype_counts.values,
        names=dtype_counts.index,
        title='Data Types Distribution'
    )
    st.plotly_chart(fig_dtypes, use_container_width=True)

# Export insights
st.subheader("📋 Key Insights")
insights = []

# Calculate key insights
churn_rate = (data[target_column].sum() / len(data) * 100)
insights.append(f"Overall churn rate: {churn_rate:.1f}%")

if numeric_cols:
    for col in numeric_cols[:2]:
        churned_mean = data[data[target_column] == 1][col].mean()
        not_churned_mean = data[data[target_column] == 0][col].mean()
        if churned_mean > not_churned_mean:
            insights.append(f"Churned customers have higher average {col} ({churned_mean:.2f} vs {not_churned_mean:.2f})")
        else:
            insights.append(f"Churned customers have lower average {col} ({churned_mean:.2f} vs {not_churned_mean:.2f})")

for insight in insights:
    st.write(f"• {insight}")
