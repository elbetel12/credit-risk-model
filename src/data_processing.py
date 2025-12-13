# src/data_processing.py
# USE THIS VERSION - Best of both worlds

import pandas as pd
import numpy as np
import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. AGGREGATE FEATURES FUNCTION (Keep this separate)
def create_aggregate_features(df):
    """Create customer-level summary features"""
    logger.info("Creating aggregate features...")
    
    # Group by customer
    agg_df = df.groupby('CustomerId').agg({
        'Amount': ['sum', 'mean', 'std', 'count'],
        'TransactionStartTime': ['min', 'max']
    }).reset_index()
    
    # Flatten column names
    agg_df.columns = [
        'CustomerId',
        'total_amount', 'avg_amount', 'std_amount', 'transaction_count',
        'first_transaction', 'last_transaction'
    ]
    
    return agg_df

# 2. TIME FEATURES FUNCTION (Keep this separate)
def extract_time_features(df):
    """Extract time-based features"""
    df = df.copy()
    
    # Convert to datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    # Extract components
    df['transaction_hour'] = df['TransactionStartTime'].dt.hour
    df['transaction_day'] = df['TransactionStartTime'].dt.day
    df['transaction_dayofweek'] = df['TransactionStartTime'].dt.dayofweek
    
    # Create patterns
    df['is_weekend'] = df['transaction_dayofweek'].isin([5, 6]).astype(int)
    df['is_business_hours'] = ((df['transaction_hour'] >= 9) & 
                               (df['transaction_hour'] <= 17)).astype(int)
    
    return df

# 3. SKLEARN PIPELINE (For encoding, missing values, scaling)
def create_feature_pipeline():
    """Create sklearn pipeline for automated processing"""
    logger.info("Creating feature pipeline...")
    
    # Define columns
    numerical_cols = ['Amount', 'Value', 'transaction_hour']
    categorical_cols = ['ChannelId', 'ProductCategory']
    
    # Numerical pipeline (handles missing + scaling)
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline (handles missing + encoding)
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])
    
    return preprocessor

# 4. MAIN PROCESSING FUNCTION
def process_data(df):
    """Complete data processing workflow"""
    logger.info("Starting data processing...")
    
    # Step 1: Time features
    df_with_time = extract_time_features(df)
    logger.info(f"After time features: {df_with_time.shape}")
    
    # Step 2: Aggregate features
    agg_features = create_aggregate_features(df_with_time)
    logger.info(f"Aggregate features: {agg_features.shape}")
    
    # Step 3: Create pipeline
    pipeline = create_feature_pipeline()
    
    # Step 4: Prepare for pipeline
    # Select only columns the pipeline expects
    features_for_pipeline = df_with_time[['Amount', 'Value', 'transaction_hour',
                                         'ChannelId', 'ProductCategory']]
    
    # Step 5: Transform
    X_processed = pipeline.fit_transform(features_for_pipeline)
    logger.info(f"Processed features shape: {X_processed.shape}")
    
    return {
        'agg_features': agg_features,
        'processed_X': X_processed,
        'pipeline': pipeline
    }

# In src/data_processing.py, add:

from woe_transformer import create_woe_features

def full_feature_engineering_pipeline(df):
    """
    Complete feature engineering pipeline including WoE
    """
    # ... existing feature engineering code ...
    
    # After creating basic features, add WoE transformation
    if 'is_high_risk' in df.columns:  # Only if target exists
        logger.info("Applying WoE transformation...")
        df_woe, woe_model, iv_values = create_woe_features(df)
        
        # Save WoE model for later use
        import joblib
        joblib.dump(woe_model, '../models/woe_transformer.pkl')
        
        return df_woe
    else:
        logger.info("Target variable not found. Skipping WoE transformation.")
        return df
    

def main():
    """Main execution"""
    # Load data
    df = pd.read_csv('../data/raw/data.csv')
    
    # Process data
    results = process_data(df)
    
    # Save
    results['agg_features'].to_csv('../data/processed/customer_features.csv', index=False)
    logger.info("Saved features to data/processed/customer_features.csv")
    
    return results

if __name__ == "__main__":
    main()