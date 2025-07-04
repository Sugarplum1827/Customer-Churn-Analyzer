import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import streamlit as st
import traceback


class DataProcessor:

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputers = {}

    def preprocess_data(self, data, target_column):
        """
        Full preprocessing pipeline: handle missing values, encode, scale.
        """
        try:
            processed_data = data.copy()

            # 1. Remove duplicates
            initial_rows = len(processed_data)
            processed_data = processed_data.drop_duplicates()
            removed = initial_rows - len(processed_data)
            if removed > 0:
                st.info(f"Removed {removed} duplicate rows")

            # 2. Handle missing values
            processed_data = self._handle_missing_values(processed_data)

            # 3. Validate and encode the target column FIRST
            processed_data = self._prepare_target_variable(
                processed_data, target_column)

            # 4. Define feature columns
            feature_columns = [
                col for col in processed_data.columns if col != target_column
            ]

            # 5. Encode categorical variables (excluding target)
            processed_data = self._encode_categorical_variables(
                processed_data, feature_columns, target_column)

            # 6. Update feature columns post-encoding
            feature_columns = [
                col for col in processed_data.columns if col != target_column
            ]

            # 7. Scale numeric features
            numeric_cols = processed_data[feature_columns].select_dtypes(
                include=[np.number]).columns
            if len(numeric_cols) > 0:
                processed_data[numeric_cols] = self.scaler.fit_transform(
                    processed_data[numeric_cols])

            # Return
            return processed_data, feature_columns

        except Exception as e:
            st.error("❌ Error in data preprocessing:")
            st.code(traceback.format_exc())
            raise e

    def _handle_missing_values(self, data):
        """
        Impute missing values with median (numeric) or mode (categorical)
        """
        missing_counts = data.isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0]

        if len(cols_with_missing) > 0:
            st.info(
                f"Handling missing values in {len(cols_with_missing)} columns")

            for col in cols_with_missing.index:
                if data[col].dtype in ['object', 'category']:
                    imputer = SimpleImputer(strategy='most_frequent')
                else:
                    imputer = SimpleImputer(strategy='median')

                data[col] = imputer.fit_transform(data[[col]]).ravel()
                self.imputers[col] = imputer

        return data

    def _encode_categorical_variables(self, data, feature_columns,
                                      target_column):
        """
        Encode categorical variables using one-hot (low cardinality) or label encoding (high)
        """
        cat_cols = [
            col for col in data[feature_columns].select_dtypes(
                include=['object', 'category']).columns if col != target_column
        ]

        if len(cat_cols) > 0:
            st.info(f"Encoding {len(cat_cols)} categorical variables")

            for col in cat_cols:
                n_unique = data[col].nunique()

                if n_unique <= 10:
                    # One-hot encoding
                    dummies = pd.get_dummies(data[col],
                                             prefix=col,
                                             drop_first=True)
                    data = pd.concat([data.drop(columns=col), dummies], axis=1)
                else:
                    # Label encoding
                    le = LabelEncoder()
                    data[col] = le.fit_transform(data[col].astype(str))
                    self.label_encoders[col] = le

        return data

    def _prepare_target_variable(self, data, target_column):
        """
        Validate and encode target for binary classification (0/1)
        """
        values = data[target_column].unique()

        if len(values) != 2:
            st.error(
                f"❌ The selected target column '{target_column}' has {len(values)} unique values."
            )
            if len(values) <= 10:
                st.dataframe(
                    data[target_column].value_counts().to_frame("Count"))
                st.info(
                    "💡 You might need to map these values to binary (e.g., 'Yes'/'No' → 1/0)."
                )
            else:
                st.write(f"Sample values: {list(values[:10])}")
                st.info(
                    "💡 Too many unique values — this is not suitable for binary churn prediction."
                )
            raise ValueError(
                f"Target variable must have exactly 2 unique values, found {len(values)}"
            )

        # Binary encoding
        if data[target_column].dtype == 'object':
            le = LabelEncoder()
            data[target_column] = le.fit_transform(data[target_column])
            self.label_encoders[target_column] = le
        else:
            sorted_vals = sorted(values)
            if sorted_vals != [0, 1]:
                data[target_column] = data[target_column].map({
                    sorted_vals[0]: 0,
                    sorted_vals[1]: 1
                })

        return data

    def transform_new_data(self, new_data, feature_columns):
        """
        Transform unseen data using stored scalers/encoders (for prediction phase)
        """
        data = new_data.copy()

        # Impute
        for col, imputer in self.imputers.items():
            if col in data.columns:
                data[col] = imputer.transform(data[[col]]).ravel()

        # Encode
        for col, encoder in self.label_encoders.items():
            if col in data.columns:
                data[col] = encoder.transform(data[col].astype(str))

        # Scale
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            data[numeric_cols] = self.scaler.transform(data[numeric_cols])

        return data[feature_columns]
