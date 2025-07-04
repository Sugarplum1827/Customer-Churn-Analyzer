import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.title("🔮 Churn Predictions")

if not st.session_state.models:
    st.warning("⚠️ Please train models first in the Model Training section")
    st.stop()

models = st.session_state.models
feature_columns = st.session_state.feature_columns

# Prediction options
st.subheader("🎯 Prediction Options")

prediction_type = st.radio(
    "Select prediction type:",
    ["Single Customer", "Batch Prediction", "Risk Segmentation"]
)

# Model selection for predictions
selected_model = st.selectbox(
    "Select model for predictions:",
    list(models.keys())
)

model = models[selected_model]

if prediction_type == "Single Customer":
    st.subheader("👤 Single Customer Prediction")
    
    # Create input form for single customer
    st.write("Enter customer information:")
    
    input_data = {}
    
    # Get sample data for reference
    if st.session_state.processed_data is not None:
        sample_data = st.session_state.processed_data[feature_columns]
        
        # Create input fields based on feature types
        cols = st.columns(2)
        
        for i, feature in enumerate(feature_columns):
            with cols[i % 2]:
                feature_values = sample_data[feature]
                
                if feature_values.dtype in ['int64', 'float64']:
                    # Numerical feature
                    min_val = float(feature_values.min())
                    max_val = float(feature_values.max())
                    mean_val = float(feature_values.mean())
                    
                    input_data[feature] = st.number_input(
                        f"{feature}",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        key=f"input_{feature}"
                    )
                else:
                    # Categorical feature (after encoding, will be numerical)
                    unique_vals = feature_values.unique()
                    input_data[feature] = st.selectbox(
                        f"{feature}",
                        options=unique_vals,
                        key=f"input_{feature}"
                    )
        
        if st.button("🔍 Predict Churn Risk", type="primary"):
            try:
                # Prepare input for prediction
                input_df = pd.DataFrame([input_data])
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                prediction_proba = model.predict_proba(input_df)[0]
                
                churn_probability = prediction_proba[1] * 100
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Prediction", "Will Churn" if prediction == 1 else "Will Stay")
                
                with col2:
                    st.metric("Churn Probability", f"{churn_probability:.1f}%")
                
                with col3:
                    risk_level = "High" if churn_probability > 70 else "Medium" if churn_probability > 30 else "Low"
                    st.metric("Risk Level", risk_level)
                
                # Risk gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = churn_probability,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Churn Risk Score"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Recommendations
                st.subheader("💡 Recommendations")
                if churn_probability > 70:
                    st.error("🚨 High churn risk! Immediate action recommended:")
                    st.write("• Contact customer for retention offer")
                    st.write("• Review customer satisfaction")
                    st.write("• Consider personalized discounts")
                elif churn_probability > 30:
                    st.warning("⚠️ Medium churn risk. Monitor closely:")
                    st.write("• Proactive engagement recommended")
                    st.write("• Check service quality metrics")
                    st.write("• Consider loyalty programs")
                else:
                    st.success("✅ Low churn risk. Customer likely to stay:")
                    st.write("• Continue current service level")
                    st.write("• Opportunity for upselling")
                    st.write("• Monitor for satisfaction")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

elif prediction_type == "Batch Prediction":
    st.subheader("📊 Batch Prediction")
    
    # File upload for batch prediction
    uploaded_file = st.file_uploader(
        "Upload CSV file for batch prediction",
        type=['csv'],
        help="Upload a CSV file with the same features as training data"
    )
    
    if uploaded_file is not None:
        try:
            # Load batch data
            batch_data = pd.read_csv(uploaded_file)
            
            st.write(f"Loaded {len(batch_data)} customers for prediction")
            st.dataframe(batch_data.head(), use_container_width=True)
            
            # Check if all required features are present
            missing_features = set(feature_columns) - set(batch_data.columns)
            if missing_features:
                st.error(f"Missing features: {missing_features}")
                st.stop()
            
            if st.button("🚀 Generate Predictions", type="primary"):
                with st.spinner("Generating predictions..."):
                    # Prepare data
                    X_batch = batch_data[feature_columns]
                    
                    # Make predictions
                    predictions = model.predict(X_batch)
                    prediction_probas = model.predict_proba(X_batch)[:, 1]
                    
                    # Add predictions to dataframe
                    results_df = batch_data.copy()
                    results_df['Churn_Prediction'] = predictions
                    results_df['Churn_Probability'] = prediction_probas
                    results_df['Risk_Level'] = pd.cut(
                        prediction_probas, 
                        bins=[0, 0.3, 0.7, 1.0], 
                        labels=['Low', 'Medium', 'High']
                    )
                    
                    # Display results
                    st.success(f"✅ Generated predictions for {len(batch_data)} customers")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Customers", len(results_df))
                    with col2:
                        predicted_churners = predictions.sum()
                        st.metric("Predicted Churners", predicted_churners)
                    with col3:
                        churn_rate = (predicted_churners / len(results_df) * 100)
                        st.metric("Predicted Churn Rate", f"{churn_rate:.1f}%")
                    with col4:
                        high_risk = (results_df['Risk_Level'] == 'High').sum()
                        st.metric("High Risk Customers", high_risk)
                    
                    # Risk distribution
                    risk_dist = results_df['Risk_Level'].value_counts()
                    fig_risk = px.pie(
                        values=risk_dist.values,
                        names=risk_dist.index,
                        title='Risk Level Distribution',
                        color_discrete_map={'Low': 'green', 'Medium': 'yellow', 'High': 'red'}
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)
                    
                    # Probability distribution
                    fig_prob = px.histogram(
                        results_df,
                        x='Churn_Probability',
                        nbins=20,
                        title='Churn Probability Distribution',
                        labels={'Churn_Probability': 'Churn Probability', 'count': 'Number of Customers'}
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)
                    
                    # Display results table
                    st.subheader("📋 Detailed Results")
                    
                    # Filter options
                    col1, col2 = st.columns(2)
                    with col1:
                        filter_risk = st.multiselect(
                            "Filter by risk level:",
                            ['Low', 'Medium', 'High'],
                            default=['High', 'Medium']
                        )
                    with col2:
                        min_prob = st.slider("Minimum churn probability:", 0.0, 1.0, 0.0)
                    
                    # Apply filters
                    filtered_df = results_df[
                        (results_df['Risk_Level'].isin(filter_risk)) &
                        (results_df['Churn_Probability'] >= min_prob)
                    ].sort_values('Churn_Probability', ascending=False)
                    
                    st.dataframe(filtered_df, use_container_width=True)
                    
                    # Download predictions
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name="churn_predictions.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing batch data: {str(e)}")

elif prediction_type == "Risk Segmentation":
    st.subheader("📊 Customer Risk Segmentation")
    
    if st.session_state.processed_data is not None:
        # Use existing processed data for segmentation
        data = st.session_state.processed_data
        X = data[feature_columns]
        
        if st.button("🎯 Generate Risk Segmentation", type="primary"):
            with st.spinner("Analyzing customer segments..."):
                # Generate predictions for all customers
                predictions = model.predict(X)
                prediction_probas = model.predict_proba(X)[:, 1]
                
                # Create risk segments
                risk_segments = pd.cut(
                    prediction_probas,
                    bins=[0, 0.3, 0.7, 1.0],
                    labels=['Low Risk', 'Medium Risk', 'High Risk']
                )
                
                # Segment analysis
                segment_analysis = pd.DataFrame({
                    'Risk_Segment': risk_segments,
                    'Churn_Probability': prediction_probas,
                    'Actual_Churn': data[st.session_state.target_column]
                })
                
                # Summary by segment
                segment_summary = segment_analysis.groupby('Risk_Segment').agg({
                    'Churn_Probability': ['count', 'mean'],
                    'Actual_Churn': 'mean'
                }).round(3)
                
                segment_summary.columns = ['Customer_Count', 'Avg_Churn_Probability', 'Actual_Churn_Rate']
                segment_summary = segment_summary.reset_index()
                
                # Display segment summary
                st.subheader("📊 Segment Summary")
                st.dataframe(segment_summary, use_container_width=True)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Customer distribution by segment
                    fig_dist = px.bar(
                        segment_summary,
                        x='Risk_Segment',
                        y='Customer_Count',
                        title='Customer Distribution by Risk Segment',
                        color='Risk_Segment',
                        color_discrete_map={'Low Risk': 'green', 'Medium Risk': 'yellow', 'High Risk': 'red'}
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                with col2:
                    # Churn rates comparison
                    comparison_data = segment_summary.melt(
                        id_vars=['Risk_Segment'],
                        value_vars=['Avg_Churn_Probability', 'Actual_Churn_Rate'],
                        var_name='Metric',
                        value_name='Rate'
                    )
                    
                    fig_comparison = px.bar(
                        comparison_data,
                        x='Risk_Segment',
                        y='Rate',
                        color='Metric',
                        barmode='group',
                        title='Predicted vs Actual Churn Rates'
                    )
                    st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Actionable insights
                st.subheader("💡 Actionable Insights")
                
                high_risk_count = segment_summary[segment_summary['Risk_Segment'] == 'High Risk']['Customer_Count'].iloc[0]
                medium_risk_count = segment_summary[segment_summary['Risk_Segment'] == 'Medium Risk']['Customer_Count'].iloc[0]
                
                insights = [
                    f"🚨 {high_risk_count} customers are at high risk of churning",
                    f"⚠️ {medium_risk_count} customers need attention to prevent churn",
                    f"💰 Focus retention efforts on high-risk segment for maximum ROI",
                    f"📈 Monitor medium-risk customers to prevent escalation"
                ]
                
                for insight in insights:
                    st.write(insight)
                
                # Segment-specific recommendations
                st.subheader("🎯 Segment-Specific Recommendations")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**High Risk Customers:**")
                    st.write("• Immediate retention calls")
                    st.write("• Special discount offers")
                    st.write("• Executive escalation")
                    st.write("• Service quality review")
                
                with col2:
                    st.write("**Medium Risk Customers:**")
                    st.write("• Proactive engagement")
                    st.write("• Satisfaction surveys")
                    st.write("• Loyalty program enrollment")
                    st.write("• Usage pattern analysis")
                
                with col3:
                    st.write("**Low Risk Customers:**")
                    st.write("• Maintain service quality")
                    st.write("• Upselling opportunities")
                    st.write("• Referral programs")
                    st.write("• Routine satisfaction checks")
    
    else:
        st.info("No processed data available for segmentation analysis")
