import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

class TransactionProcessor:
    def __init__(self):
        self.processed_data = None
        
    def convert_transactions_to_churn_data(self, transactions_df, observation_date=None, churn_days=30):
        """
        Convert transaction data to customer-level churn dataset
        
        Parameters:
        - transactions_df: DataFrame with transaction data
        - observation_date: Date to use as cutoff (default: 30 days before max date)
        - churn_days: Number of days to define churn (default: 30)
        """
        try:
            # Parse transaction dates
            transactions_df['Transaction Date'] = pd.to_datetime(transactions_df['Transaction Date'])
            
            # Set observation date if not provided
            max_date = transactions_df['Transaction Date'].max()
            if observation_date is None:
                observation_date = max_date - timedelta(days=churn_days)
            else:
                observation_date = pd.to_datetime(observation_date)
            
            # Split data into before and after observation date
            historical_data = transactions_df[transactions_df['Transaction Date'] <= observation_date]
            future_data = transactions_df[transactions_df['Transaction Date'] > observation_date]
            
            # Create customer features from historical data
            customer_features = self._create_customer_features(historical_data, observation_date)
            
            # Define churn based on future activity
            active_customers_future = set(future_data['Name'].unique())
            customer_features['churn'] = customer_features.index.map(
                lambda x: 0 if x in active_customers_future else 1
            )
            
            # Reset index to make customer name a column
            customer_features = customer_features.reset_index()
            
            st.success(f"✅ Converted {len(transactions_df)} transactions into {len(customer_features)} customer records")
            st.info(f"📅 Using observation date: {observation_date.strftime('%Y-%m-%d')}")
            st.info(f"🔍 Churn definition: No activity for {churn_days} days after observation date")
            
            return customer_features
            
        except Exception as e:
            st.error(f"Error processing transaction data: {str(e)}")
            raise e
    
    def _create_customer_features(self, historical_data, observation_date):
        """
        Create customer-level features from transaction history
        """
        # Group by customer
        customer_groups = historical_data.groupby('Name')
        
        features = {}
        
        # Transaction count features
        features['total_transactions'] = customer_groups.size()
        features['first_transaction_date'] = customer_groups['Transaction Date'].min()
        features['last_transaction_date'] = customer_groups['Transaction Date'].max()
        
        # Calculate tenure (days from first transaction to observation date)
        features_df = pd.DataFrame(features)
        features_df['tenure_days'] = (observation_date - features_df['first_transaction_date']).dt.days
        features_df['days_since_last_transaction'] = (observation_date - features_df['last_transaction_date']).dt.days
        
        # Transaction frequency features
        features_df['avg_days_between_transactions'] = features_df['tenure_days'] / features_df['total_transactions']
        
        # Product features
        product_features = self._create_product_features(historical_data)
        features_df = features_df.join(product_features)
        
        # Email domain features
        email_features = self._create_email_features(historical_data)
        features_df = features_df.join(email_features)
        
        # Recency, Frequency, Monetary-like features
        rfm_features = self._create_rfm_features(historical_data, observation_date)
        features_df = features_df.join(rfm_features)
        
        # Drop date columns as they're not needed for ML
        features_df = features_df.drop(['first_transaction_date', 'last_transaction_date'], axis=1)
        
        # Fill any missing values
        features_df = features_df.fillna(0)
        
        return features_df
    
    def _create_product_features(self, data):
        """
        Create product-related features
        """
        product_data = []
        
        for name, group in data.groupby('Name'):
            # Parse products (handle comma-separated products)
            all_products = []
            for product_str in group['Product']:
                if pd.notna(product_str):
                    products = [p.strip() for p in str(product_str).split(',')]
                    all_products.extend(products)
            
            # Count unique products
            unique_products = set(all_products)
            
            product_data.append({
                'Name': name,
                'unique_products_count': len(unique_products),
                'total_product_purchases': len(all_products),
                'avg_products_per_transaction': len(all_products) / len(group) if len(group) > 0 else 0
            })
        
        product_df = pd.DataFrame(product_data).set_index('Name')
        return product_df
    
    def _create_email_features(self, data):
        """
        Create email domain-related features
        """
        email_data = []
        
        for name, group in data.groupby('Name'):
            email = group['Email'].iloc[0]  # Take first email (should be same for all)
            
            if pd.notna(email) and '@' in str(email):
                domain = str(email).split('@')[1].lower()
                
                # Categorize email domains
                is_gmail = 1 if 'gmail' in domain else 0
                is_yahoo = 1 if 'yahoo' in domain else 0
                is_corporate = 1 if domain not in ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'rediffmail.com'] else 0
                
                email_data.append({
                    'Name': name,
                    'is_gmail': is_gmail,
                    'is_yahoo': is_yahoo,
                    'is_corporate_email': is_corporate
                })
            else:
                email_data.append({
                    'Name': name,
                    'is_gmail': 0,
                    'is_yahoo': 0,
                    'is_corporate_email': 0
                })
        
        email_df = pd.DataFrame(email_data).set_index('Name')
        return email_df
    
    def _create_rfm_features(self, data, observation_date):
        """
        Create RFM-like features (Recency, Frequency, Monetary proxy)
        """
        rfm_data = []
        
        for name, group in data.groupby('Name'):
            # Recency: days since last transaction
            last_transaction = group['Transaction Date'].max()
            recency = (observation_date - last_transaction).days
            
            # Frequency: number of transactions
            frequency = len(group)
            
            # Monetary proxy: unique products (since we don't have monetary values)
            all_products = []
            for product_str in group['Product']:
                if pd.notna(product_str):
                    products = [p.strip() for p in str(product_str).split(',')]
                    all_products.extend(products)
            monetary_proxy = len(set(all_products))
            
            # Calculate transaction pattern features
            date_range = (group['Transaction Date'].max() - group['Transaction Date'].min()).days
            transaction_span = max(date_range, 1)  # Avoid division by zero
            
            rfm_data.append({
                'Name': name,
                'recency_days': recency,
                'frequency_count': frequency,
                'monetary_proxy': monetary_proxy,
                'transaction_span_days': transaction_span,
                'transaction_density': frequency / transaction_span  # transactions per day
            })
        
        rfm_df = pd.DataFrame(rfm_data).set_index('Name')
        return rfm_df