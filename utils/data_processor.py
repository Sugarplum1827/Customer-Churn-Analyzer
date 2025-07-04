import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import streamlit as st

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputers = {}
        
    def preprocess_data(self, data, target_column):
        """
        Comprehensive data preprocessing pipeline
        """
        try:
            # Make a copy to avoid modifying original data
            processed_data = data.copy()
            
            # Remove duplicates
            initial_rows = len(processed_data)
            processed_data = processed_data.drop_duplicates()
            duplicates_removed = initial_rows - len(processed_data)
            
            if duplicates_removed > 0:
                st.info(f"Removed {duplicates_removed} duplicate rows")
            
            # Handle missing values
            processed_data = self._handle_missing_values(processed_data)
            
            # Separate features and target
            feature_columns = [col for col in processed_data.columns if col != target_column]
            
            # Encode categorical variables
            processed_data = self._encode_categorical_variables(processed_data, feature_columns)
            
            # Handle target variable
            processed_data = self._prepare_target_variable(processed_data, target_column)
            
            # Feature scaling for numerical variables
            numerical_columns = processed_data[feature_columns].select_dtypes(include=[np.number]).columns
            if len(numerical_columns) > 0:
                processed_data[numerical_columns] = self.scaler.fit_transform(processed_data[numerical_columns])
            
            # Update feature columns after encoding
            final_feature_columns = [col for col in processed_data.columns if col != target_column]
            
            return processed_data, final_feature_columns
            
        except Exception as e:
            st.error(f"Error in data preprocessing: {str(e)}")
            raise e
    
    def _handle_missing_values(self, data):
        """
        Handle missing values in the dataset
        """
        missing_counts = data.isnull().sum()
        columns_with_missing = missing_counts[missing_counts > 0]
        
        if len(columns_with_missing) > 0:
            st.info(f"Handling missing values in {len(columns_with_missing)} columns")
            
            for column in columns_with_missing.index:
                if data[column].dtype in ['object', 'category']:
                    # For categorical variables, use mode
                    imputer = SimpleImputer(strategy='most_frequent')
                    data[column] = imputer.fit_transform(data[[column]]).ravel()
                else:
                    # For numerical variables, use median
                    imputer = SimpleImputer(strategy='median')
                    data[column] = imputer.fit_transform(data[[column]]).ravel()
                
                self.imputers[column] = imputer
        
        return data
    
    def _encode_categorical_variables(self, data, feature_columns):
        """
        Encode categorical variables using label encoding and one-hot encoding
        """
        categorical_columns = data[feature_columns].select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_columns) > 0:
            st.info(f"Encoding {len(categorical_columns)} categorical variables")
            
            for column in categorical_columns:
                unique_values = data[column].nunique()
                
                if unique_values <= 10:  # Use one-hot encoding for low cardinality
                    # One-hot encoding
                    dummies = pd.get_dummies(data[column], prefix=column, drop_first=True)
                    data = pd.concat([data, dummies], axis=1)
                    data = data.drop(column, axis=1)
                else:  # Use label encoding for high cardinality
                    # Label encoding
                    le = LabelEncoder()
                    data[column] = le.fit_transform(data[column].astype(str))
                    self.label_encoders[column] = le
        
        return data
    
    def _prepare_target_variable(self, data, target_column):
        """
        Prepare target variable for binary classification
        """
        target_values = data[target_column].unique()
        
        # Convert target to binary if needed
        if len(target_values) == 2:
            if data[target_column].dtype == 'object':
                # Convert string labels to 0/1
                le = LabelEncoder()
                data[target_column] = le.fit_transform(data[target_column])
                self.label_encoders[target_column] = le
            else:
                # Ensure values are 0 and 1
                unique_vals = sorted(data[target_column].unique())
                if unique_vals != [0, 1]:
                    data[target_column] = data[target_column].map({unique_vals[0]: 0, unique_vals[1]: 1})
        else:
            raise ValueError(f"Target variable must have exactly 2 unique values, found {len(target_values)}")
        
        return data
    
    def transform_new_data(self, new_data, feature_columns):
        """
        Transform new data using fitted preprocessors
        """
        processed_data = new_data.copy()
        
        # Handle missing values
        for column, imputer in self.imputers.items():
            if column in processed_data.columns:
                processed_data[column] = imputer.transform(processed_data[[column]]).ravel()
        
        # Encode categorical variables
        for column, encoder in self.label_encoders.items():
            if column in processed_data.columns:
                processed_data[column] = encoder.transform(processed_data[column].astype(str))
        
        # Scale numerical features
        numerical_columns = processed_data.select_dtypes(include=[np.number]).columns
        if len(numerical_columns) > 0:
            processed_data[numerical_columns] = self.scaler.transform(processed_data[numerical_columns])
        
        return processed_data[feature_columns]
