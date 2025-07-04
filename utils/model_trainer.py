import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import xgboost as xgb
import streamlit as st

class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {
            'logistic': LogisticRegression(random_state=random_state, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=random_state, n_estimators=100),
            'xgboost': xgb.XGBClassifier(random_state=random_state, eval_metric='logloss'),
            'svm': SVC(random_state=random_state, probability=True),
            'gradient_boosting': GradientBoostingClassifier(random_state=random_state)
        }
    
    def train_model(self, model_type, X_train, y_train, X_test, y_test, cv_folds=None):
        """
        Train a specific model and return model with evaluation metrics
        """
        try:
            # Get the model
            model = self.models[model_type].fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='binary'),
                'recall': recall_score(y_test, y_pred, average='binary'),
                'f1_score': f1_score(y_test, y_pred, average='binary'),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred)
            }
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            metrics['fpr'] = fpr
            metrics['tpr'] = tpr
            
            # Cross-validation if requested
            if cv_folds:
                cv_scores = cross_val_score(
                    self.models[model_type], X_train, y_train, 
                    cv=cv_folds, scoring='accuracy'
                )
                metrics['cv_mean'] = cv_scores.mean()
                metrics['cv_std'] = cv_scores.std()
            
            return model, metrics
            
        except Exception as e:
            st.error(f"Error training {model_type} model: {str(e)}")
            raise e
    
    def train_multiple_models(self, model_types, X_train, y_train, X_test, y_test, cv_folds=None):
        """
        Train multiple models and return comparison results
        """
        results = {}
        models = {}
        
        for model_type in model_types:
            st.write(f"Training {model_type}...")
            model, metrics = self.train_model(
                model_type, X_train, y_train, X_test, y_test, cv_folds
            )
            
            models[model_type] = model
            results[model_type] = metrics
        
        return models, results
    
    def get_model_config(self, model_type):
        """
        Get model configuration for hyperparameter tuning
        """
        configs = {
            'logistic': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }
        
        return configs.get(model_type, {})
    
    def evaluate_model_performance(self, model, X_test, y_test):
        """
        Comprehensive model evaluation
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate detailed metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1_score': f1_score(y_test, y_pred, average='binary'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # ROC curve data
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        metrics['roc_data'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
        
        # Prediction distribution
        metrics['prediction_distribution'] = {
            'predicted_positive': (y_pred == 1).sum(),
            'predicted_negative': (y_pred == 0).sum(),
            'actual_positive': (y_test == 1).sum(),
            'actual_negative': (y_test == 0).sum()
        }
        
        return metrics
    
    def get_feature_importance(self, model, feature_names):
        """
        Extract feature importance from model if available
        """
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(model.coef_[0])
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        else:
            return None
    
    def predict_churn_probability(self, model, X):
        """
        Predict churn probability for new data
        """
        try:
            probabilities = model.predict_proba(X)[:, 1]
            predictions = model.predict(X)
            
            return predictions, probabilities
            
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")
            raise e
    
    def calculate_business_metrics(self, y_true, y_pred, y_pred_proba, 
                                 customer_value=100, retention_cost=20):
        """
        Calculate business impact metrics
        """
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Business metrics
        total_customers = len(y_true)
        actual_churners = (y_true == 1).sum()
        predicted_churners = (y_pred == 1).sum()
        
        # Cost calculations
        false_positive_cost = fp * retention_cost  # Cost of unnecessary retention efforts
        false_negative_cost = fn * customer_value  # Cost of lost customers
        true_positive_benefit = tp * customer_value  # Value of prevented churn
        
        total_cost = false_positive_cost + false_negative_cost
        total_benefit = true_positive_benefit
        net_benefit = total_benefit - total_cost
        
        metrics = {
            'total_customers': total_customers,
            'actual_churners': actual_churners,
            'predicted_churners': predicted_churners,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'false_positive_cost': false_positive_cost,
            'false_negative_cost': false_negative_cost,
            'true_positive_benefit': true_positive_benefit,
            'total_cost': total_cost,
            'total_benefit': total_benefit,
            'net_benefit': net_benefit,
            'roi': (net_benefit / max(total_cost, 1)) * 100  # Avoid division by zero
        }
        
        return metrics
