import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns

class ChurnVisualizer:
    def __init__(self):
        self.color_palette = {
            'churn': '#e74c3c',
            'no_churn': '#2ecc71',
            'primary': '#3498db',
            'secondary': '#9b59b6'
        }
    
    def plot_churn_distribution(self, data, target_column):
        """
        Plot the distribution of churn vs non-churn customers
        """
        churn_counts = data[target_column].value_counts()
        labels = ['No Churn', 'Churn'] if 0 in churn_counts.index else churn_counts.index
        
        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=churn_counts.values,
                marker_color=[self.color_palette['no_churn'], self.color_palette['churn']],
                text=churn_counts.values,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Customer Churn Distribution',
            xaxis_title='Churn Status',
            yaxis_title='Number of Customers',
            height=400
        )
        
        return fig
    
    def plot_numerical_features_boxplot(self, data, numeric_columns, target_column):
        """
        Create box plots for numerical features by churn status
        """
        n_cols = min(2, len(numeric_columns))
        n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=numeric_columns,
            vertical_spacing=0.08
        )
        
        for i, col in enumerate(numeric_columns):
            row = i // n_cols + 1
            col_pos = i % n_cols + 1
            
            # Data for non-churned customers
            no_churn_data = data[data[target_column] == 0][col]
            # Data for churned customers
            churn_data = data[data[target_column] == 1][col]
            
            fig.add_trace(
                go.Box(
                    y=no_churn_data,
                    name='No Churn',
                    marker_color=self.color_palette['no_churn'],
                    showlegend=(i == 0)
                ),
                row=row, col=col_pos
            )
            
            fig.add_trace(
                go.Box(
                    y=churn_data,
                    name='Churn',
                    marker_color=self.color_palette['churn'],
                    showlegend=(i == 0)
                ),
                row=row, col=col_pos
            )
        
        fig.update_layout(
            title='Numerical Features Distribution by Churn Status',
            height=300 * n_rows,
            boxmode='group'
        )
        
        return fig
    
    def plot_correlation_heatmap(self, data):
        """
        Create a correlation heatmap
        """
        corr_matrix = data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Feature Correlation Heatmap',
            height=500,
            width=500
        )
        
        return fig
    
    def plot_categorical_churn_analysis(self, data, categorical_columns, target_column):
        """
        Create bar plots showing churn rates for categorical features
        """
        n_cols = min(2, len(categorical_columns))
        n_rows = (len(categorical_columns) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=categorical_columns,
            vertical_spacing=0.1
        )
        
        for i, col in enumerate(categorical_columns):
            row = i // n_cols + 1
            col_pos = i % n_cols + 1
            
            # Calculate churn rate by category
            churn_by_category = data.groupby(col)[target_column].agg(['count', 'sum']).reset_index()
            churn_by_category['churn_rate'] = (churn_by_category['sum'] / churn_by_category['count'] * 100).round(2)
            
            fig.add_trace(
                go.Bar(
                    x=churn_by_category[col],
                    y=churn_by_category['churn_rate'],
                    name=f'{col} Churn Rate',
                    marker_color=self.color_palette['primary'],
                    showlegend=False,
                    text=churn_by_category['churn_rate'],
                    textposition='auto'
                ),
                row=row, col=col_pos
            )
        
        fig.update_layout(
            title='Churn Rate by Categorical Features',
            height=300 * n_rows
        )
        
        # Update y-axis titles
        for i in range(1, n_rows + 1):
            for j in range(1, n_cols + 1):
                fig.update_yaxes(title_text="Churn Rate (%)", row=i, col=j)
        
        return fig
    
    def plot_feature_distribution(self, data, feature, target_column):
        """
        Plot feature distribution split by churn status
        """
        fig = go.Figure()
        
        # No churn distribution
        no_churn_data = data[data[target_column] == 0][feature]
        fig.add_trace(go.Histogram(
            x=no_churn_data,
            name='No Churn',
            marker_color=self.color_palette['no_churn'],
            opacity=0.7,
            nbinsx=30
        ))
        
        # Churn distribution
        churn_data = data[data[target_column] == 1][feature]
        fig.add_trace(go.Histogram(
            x=churn_data,
            name='Churn',
            marker_color=self.color_palette['churn'],
            opacity=0.7,
            nbinsx=30
        ))
        
        fig.update_layout(
            title=f'Distribution of {feature} by Churn Status',
            xaxis_title=feature,
            yaxis_title='Frequency',
            barmode='overlay',
            height=400
        )
        
        return fig
    
    def plot_model_performance_comparison(self, results_dict):
        """
        Create a comparison chart for multiple model performances
        """
        models = list(results_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [results_dict[model][metric] for model in models]
            fig.add_trace(go.Bar(
                name=metric.title(),
                x=models,
                y=values,
                text=[f'{v:.3f}' for v in values],
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Models',
            yaxis_title='Score',
            barmode='group',
            height=500
        )
        
        return fig
    
    def plot_roc_curves(self, results_dict):
        """
        Plot ROC curves for multiple models
        """
        fig = go.Figure()
        
        for model_name, metrics in results_dict.items():
            fpr = metrics['fpr']
            tpr = metrics['tpr']
            auc = metrics['roc_auc']
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_name} (AUC = {auc:.3f})',
                line=dict(width=2)
            ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))
        
        fig.update_layout(
            title='ROC Curves Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500,
            width=600
        )
        
        return fig
    
    def plot_confusion_matrix(self, cm, model_name):
        """
        Plot confusion matrix heatmap
        """
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted No Churn', 'Predicted Churn'],
            y=['Actual No Churn', 'Actual Churn'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 16},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f'Confusion Matrix - {model_name}',
            height=400,
            width=400
        )
        
        return fig
    
    def plot_feature_importance(self, importance_df, model_name, top_n=15):
        """
        Plot feature importance
        """
        # Take top N features
        top_features = importance_df.head(top_n)
        
        fig = go.Figure(go.Bar(
            x=top_features['importance'],
            y=top_features['feature'],
            orientation='h',
            marker_color=self.color_palette['primary']
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Feature Importance - {model_name}',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def plot_churn_probability_distribution(self, probabilities, actual_labels):
        """
        Plot distribution of churn probabilities
        """
        fig = go.Figure()
        
        # Probabilities for actual non-churners
        no_churn_probs = probabilities[actual_labels == 0]
        fig.add_trace(go.Histogram(
            x=no_churn_probs,
            name='Actual No Churn',
            marker_color=self.color_palette['no_churn'],
            opacity=0.7,
            nbinsx=50
        ))
        
        # Probabilities for actual churners
        churn_probs = probabilities[actual_labels == 1]
        fig.add_trace(go.Histogram(
            x=churn_probs,
            name='Actual Churn',
            marker_color=self.color_palette['churn'],
            opacity=0.7,
            nbinsx=50
        ))
        
        fig.update_layout(
            title='Distribution of Predicted Churn Probabilities',
            xaxis_title='Churn Probability',
            yaxis_title='Frequency',
            barmode='overlay',
            height=400
        )
        
        return fig
