import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.model_trainer import ModelTrainer

st.title("🤖 Model Training & Evaluation")

if st.session_state.processed_data is None:
    st.warning("⚠️ Please upload and process data first in the Data Upload section")
    st.stop()

processed_data = st.session_state.processed_data
feature_columns = st.session_state.feature_columns
target_column = st.session_state.target_column

# Model configuration
st.subheader("⚙️ Model Configuration")

col1, col2 = st.columns(2)

with col1:
    test_size = st.slider(
        "Test set size (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        help="Percentage of data to use for testing"
    )

with col2:
    random_state = st.number_input(
        "Random state",
        min_value=1,
        max_value=1000,
        value=42,
        help="Random seed for reproducibility"
    )

# Model selection
st.subheader("🎯 Model Selection")

available_models = {
    "Logistic Regression": "logistic",
    "Random Forest": "random_forest", 
    "XGBoost": "xgboost",
    "Support Vector Machine": "svm",
    "Gradient Boosting": "gradient_boosting"
}

selected_models = st.multiselect(
    "Select models to train:",
    list(available_models.keys()),
    default=["Logistic Regression", "Random Forest", "XGBoost"]
)

# Advanced settings
with st.expander("🔧 Advanced Settings"):
    cross_validation = st.checkbox("Enable Cross Validation", value=True)
    cv_folds = st.slider("CV Folds", 3, 10, 5) if cross_validation else 5
    
    feature_selection = st.checkbox("Enable Feature Selection", value=False)
    n_features = st.slider("Number of top features", 5, len(feature_columns), 
                          min(20, len(feature_columns))) if feature_selection else len(feature_columns)

# Train models
if st.button("🚀 Train Models", type="primary"):
    if not selected_models:
        st.error("Please select at least one model to train")
        st.stop()
    
    with st.spinner("Training models... This may take a few minutes."):
        try:
            # Prepare data
            X = processed_data[feature_columns]
            y = processed_data[target_column]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=random_state, stratify=y
            )
            
            # Initialize trainer
            trainer = ModelTrainer(random_state=random_state)
            
            # Train selected models
            models = {}
            results = {}
            
            progress_bar = st.progress(0)
            
            for i, model_name in enumerate(selected_models):
                model_key = available_models[model_name]
                
                st.write(f"Training {model_name}...")
                
                # Train model
                model, metrics = trainer.train_model(
                    model_key, X_train, y_train, X_test, y_test,
                    cv_folds=cv_folds if cross_validation else None
                )
                
                models[model_name] = model
                results[model_name] = metrics
                
                progress_bar.progress((i + 1) / len(selected_models))
            
            # Store in session state
            st.session_state.models = models
            st.session_state.model_results = results
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            
            st.success("✅ Model training completed!")
            
        except Exception as e:
            st.error(f"Error during model training: {str(e)}")
            st.stop()

# Display results if models are trained
if st.session_state.models:
    st.subheader("📊 Model Performance Comparison")
    
    results = st.session_state.model_results
    
    # Create comparison dataframe
    comparison_data = []
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'ROC-AUC': metrics['roc_auc']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display metrics table
    st.dataframe(comparison_df.round(4), use_container_width=True)
    
    # Metrics visualization
    fig_metrics = go.Figure()
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    for metric in metrics_to_plot:
        fig_metrics.add_trace(go.Bar(
            name=metric,
            x=comparison_df['Model'],
            y=comparison_df[metric],
            text=comparison_df[metric].round(3),
            textposition='auto'
        ))
    
    fig_metrics.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis_title='Score',
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig_metrics, use_container_width=True)
    
    # ROC Curves
    st.subheader("📈 ROC Curves")
    
    fig_roc = go.Figure()
    
    for model_name in results.keys():
        fpr = results[model_name]['fpr']
        tpr = results[model_name]['tpr']
        auc = results[model_name]['roc_auc']
        
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f'{model_name} (AUC = {auc:.3f})',
            mode='lines'
        ))
    
    # Add diagonal line
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray')
    ))
    
    fig_roc.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500
    )
    
    st.plotly_chart(fig_roc, use_container_width=True)
    
    # Confusion Matrices
    st.subheader("🔍 Confusion Matrices")
    
    cols = st.columns(min(len(results), 3))
    
    for i, (model_name, metrics) in enumerate(results.items()):
        with cols[i % 3]:
            cm = metrics['confusion_matrix']
            
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title=f'{model_name}',
                labels=dict(x="Predicted", y="Actual")
            )
            
            st.plotly_chart(fig_cm, use_container_width=True)
    
    # Feature Importance (for tree-based models)
    st.subheader("🌳 Feature Importance")
    
    models = st.session_state.models
    
    for model_name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(15)
            
            fig_importance = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title=f'Feature Importance - {model_name}',
                labels={'Importance': 'Feature Importance', 'Feature': 'Features'}
            )
            fig_importance.update_layout(height=400)
            
            st.plotly_chart(fig_importance, use_container_width=True)
    
    # Model recommendations
    st.subheader("💡 Model Recommendations")
    
    best_accuracy = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
    best_f1 = comparison_df.loc[comparison_df['F1-Score'].idxmax()]
    best_auc = comparison_df.loc[comparison_df['ROC-AUC'].idxmax()]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Best Accuracy",
            f"{best_accuracy['Accuracy']:.3f}",
            delta=f"{best_accuracy['Model']}"
        )
    
    with col2:
        st.metric(
            "Best F1-Score",
            f"{best_f1['F1-Score']:.3f}",
            delta=f"{best_f1['Model']}"
        )
    
    with col3:
        st.metric(
            "Best ROC-AUC",
            f"{best_auc['ROC-AUC']:.3f}",
            delta=f"{best_auc['Model']}"
        )
    
    # Detailed classification reports
    with st.expander("📋 Detailed Classification Reports"):
        for model_name, metrics in results.items():
            st.write(f"**{model_name} Classification Report:**")
            st.text(metrics['classification_report'])
            st.write("---")

else:
    st.info("👆 Configure and train models to see performance metrics")
