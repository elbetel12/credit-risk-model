# src/woe_manual.py
"""
Manual WoE and IV calculation - More reliable than xverse
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import logging
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_woe_iv_manual(df, feature_col, target_col='is_high_risk'):
    """
    Manual calculation of WoE and IV for a single feature
    """
    # Create a copy
    temp_df = df[[feature_col, target_col]].copy()
    
    # Handle missing values
    temp_df[feature_col] = temp_df[feature_col].fillna('MISSING')
    
    # For numerical features, bin them first
    if pd.api.types.is_numeric_dtype(temp_df[feature_col]):
        # Bin numerical features
        try:
            temp_df[f'{feature_col}_binned'], bins = pd.qcut(
                temp_df[feature_col], 
                q=5, 
                duplicates='drop', 
                retbins=True
            )
            feature_col_binned = f'{feature_col}_binned'
        except:
            # If qcut fails, use equal width bins
            temp_df[f'{feature_col}_binned'] = pd.cut(temp_df[feature_col], bins=5)
            feature_col_binned = f'{feature_col}_binned'
    else:
        # For categorical, use as is
        feature_col_binned = feature_col
    
    # Group by the feature
    grouped = temp_df.groupby(feature_col_binned).agg({
        target_col: ['count', 'sum']
    })
    
    grouped.columns = ['total', 'bad']
    grouped['good'] = grouped['total'] - grouped['bad']
    
    # Calculate percentages
    total_good = grouped['good'].sum()
    total_bad = grouped['bad'].sum()
    
    # Avoid division by zero
    if total_good == 0 or total_bad == 0:
        return None, 0
    
    grouped['pct_good'] = grouped['good'] / total_good
    grouped['pct_bad'] = grouped['bad'] / total_bad
    
    # Calculate WoE
    grouped['woe'] = np.log((grouped['pct_good'] + 1e-10) / (grouped['pct_bad'] + 1e-10))
    
    # Calculate IV component
    grouped['iv_component'] = (grouped['pct_good'] - grouped['pct_bad']) * grouped['woe']
    
    total_iv = grouped['iv_component'].sum()
    
    return grouped, total_iv

def woe_encode_feature(df, feature_col, target_col='is_high_risk'):
    """
    Apply WoE encoding to a feature
    """
    woe_table, iv = calculate_woe_iv_manual(df, feature_col, target_col)
    
    if woe_table is None:
        return df[feature_col], iv
    
    # Create mapping from bin to WoE value
    woe_mapping = woe_table['woe'].to_dict()
    
    # Apply mapping
    if pd.api.types.is_numeric_dtype(df[feature_col]):
        # For numerical, we need to map bins
        try:
            binned, _ = pd.qcut(df[feature_col], q=5, duplicates='drop', retbins=True)
        except:
            binned = pd.cut(df[feature_col], bins=5)
        
        # Map each bin to WoE value
        woe_values = binned.map(woe_mapping)
    else:
        # For categorical, map directly
        woe_values = df[feature_col].map(woe_mapping)
    
    return woe_values, iv

def create_woe_features_manual(df, target_col='is_high_risk'):
    """
    Manual WoE feature engineering
    """
    logger.info("="*60)
    logger.info("MANUAL WOE TRANSFORMATION")
    logger.info("="*60)
    
    # Make a copy
    df_prepared = df.copy()
    
    # Step 1: Prepare basic features
    if 'TransactionStartTime' in df_prepared.columns:
        df_prepared['TransactionStartTime'] = pd.to_datetime(df_prepared['TransactionStartTime'], errors='coerce')
        df_prepared['transaction_hour'] = df_prepared['TransactionStartTime'].dt.hour
        df_prepared['transaction_dayofweek'] = df_prepared['TransactionStartTime'].dt.dayofweek
        df_prepared['is_weekend'] = df_prepared['transaction_dayofweek'].isin([5, 6]).astype(int)
    
    if 'Amount' in df_prepared.columns:
        df_prepared['amount_abs'] = abs(df_prepared['Amount'])
        df_prepared['is_credit'] = (df_prepared['Amount'] < 0).astype(int)
    
    # Step 2: Create target if not exists
    if target_col not in df_prepared.columns:
        logger.warning(f"Target '{target_col}' not found. Creating proxy...")
        if 'FraudResult' in df_prepared.columns:
            df_prepared[target_col] = df_prepared['FraudResult']
        else:
            # Simple risk proxy: negative amount transactions
            if 'Amount' in df_prepared.columns:
                df_prepared[target_col] = (df_prepared['Amount'] < 0).astype(int)
            else:
                np.random.seed(42)
                df_prepared[target_col] = np.random.choice([0, 1], size=len(df_prepared), p=[0.9, 0.1])
    
    # Step 3: Select features for WoE
    feature_candidates = [
        'Amount', 'Value', 'transaction_hour', 'transaction_dayofweek',
        'is_weekend', 'amount_abs', 'is_credit',
        'ChannelId', 'ProductCategory', 'CountryCode', 'ProviderId',
        'PricingStrategy'
    ]
    
    available_features = [f for f in feature_candidates if f in df_prepared.columns]
    
    if not available_features:
        logger.error("No features available for WoE!")
        return df_prepared, pd.DataFrame(), {}
    
    logger.info(f"\nProcessing {len(available_features)} features...")
    
    # Step 4: Calculate WoE for each feature
    woe_results = {}
    iv_results = []
    
    for feature in available_features:
        try:
            woe_values, iv = woe_encode_feature(df_prepared, feature, target_col)
            df_prepared[f'{feature}_WOE'] = woe_values
            
            # Store results
            woe_results[feature] = woe_values
            iv_results.append({
                'Feature': feature,
                'IV': iv,
                'Type': 'Numerical' if pd.api.types.is_numeric_dtype(df_prepared[feature]) else 'Categorical'
            })
            
            logger.info(f"  ✓ {feature:25} IV = {iv:.4f}")
            
        except Exception as e:
            logger.warning(f"  ✗ {feature:25} Failed: {str(e)[:50]}")
    
    # Create IV dataframe
    iv_df = pd.DataFrame(iv_results).sort_values('IV', ascending=False)
    
    # Step 5: Analyze results
    if len(iv_df) > 0:
        logger.info("\n" + "="*60)
        logger.info("FEATURE IMPORTANCE ANALYSIS")
        logger.info("="*60)
        
        for _, row in iv_df.iterrows():
            strength = classify_iv_strength(row['IV'])
            logger.info(f"{row['Feature']:25} IV={row['IV']:7.4f}  ({strength:15}) [{row['Type']}]")
    
    return df_prepared, iv_df, woe_results

def classify_iv_strength(iv_value):
    """Classify IV value"""
    if iv_value < 0.02:
        return "Not predictive"
    elif iv_value < 0.1:
        return "Weak"
    elif iv_value < 0.3:
        return "Medium"
    else:
        return "Strong"

def visualize_results(iv_df, top_n=15):
    """Visualize IV results"""
    if iv_df.empty:
        logger.warning("No IV data to visualize")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Get top N features
    top_features = iv_df.head(top_n).sort_values('IV', ascending=True)
    
    # Create color map
    colors = []
    for iv in top_features['IV']:
        if iv < 0.02:
            colors.append('#FF6B6B')
        elif iv < 0.1:
            colors.append('#FFD166')
        elif iv < 0.3:
            colors.append('#06D6A0')
        else:
            colors.append('#118AB2')
    
    # Plot
    bars = plt.barh(range(len(top_features)), top_features['IV'], color=colors, edgecolor='black')
    
    # Add value labels
    for i, (idx, row) in enumerate(top_features.iterrows()):
        plt.text(row['IV'] + 0.001, i, f'{row["IV"]:.4f}', va='center', fontsize=9)
    
    # Customize
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Information Value (IV)', fontsize=12, fontweight='bold')
    plt.title(f'Top {len(top_features)} Features by Information Value', fontsize=14, fontweight='bold')
    
    # Add reference lines
    plt.axvline(x=0.02, color='#FF6B6B', linestyle='--', alpha=0.7, linewidth=1, label='Min predictive (0.02)')
    plt.axvline(x=0.1, color='#06D6A0', linestyle='--', alpha=0.7, linewidth=1, label='Medium (0.1)')
    plt.axvline(x=0.3, color='#118AB2', linestyle='--', alpha=0.7, linewidth=1, label='Strong (0.3)')
    
    plt.legend(loc='lower right')
    plt.grid(axis='x', alpha=0.2, linestyle='--')
    plt.tight_layout()
    
    # Save
    try:
        plt.savefig('../data/processed/woe_iv_results.png', dpi=150, bbox_inches='tight')
        logger.info("Saved visualization: data/processed/woe_iv_results.png")
    except Exception as e:
        logger.warning(f"Could not save plot: {e}")
    
    plt.show()

def main():
    """Main execution"""
    print("\n" + "="*70)
    print("MANUAL WOE & IV CALCULATION FOR CREDIT RISK")
    print("="*70)
    
    # Load data
    try:
        df = pd.read_csv('../data/raw/data.csv')
        logger.info(f"✓ Loaded transaction data: {df.shape}")
        logger.info(f"  Columns: {', '.join(df.columns[:5])}...")
    except FileNotFoundError:
        logger.error("❌ Raw data not found at: ../data/raw/data.csv")
        return None, None, None
    
    # Apply manual WoE transformation
    df_woe, iv_df, woe_results = create_woe_features_manual(df)
    
    # Save results
    if not df_woe.empty:
        # Save full dataset
        df_woe.to_csv('../data/processed/transactions_with_woe.csv', index=False)
        logger.info(f"\n✓ Saved full dataset with WoE features: data/processed/transactions_with_woe.csv")
        
        # Save only WoE features (for modeling)
        woe_cols = [col for col in df_woe.columns if '_WOE' in col]
        if woe_cols:
            woe_df = df_woe[['CustomerId', 'is_high_risk'] + woe_cols]
            woe_df.to_csv('../data/processed/woe_features_only.csv', index=False)
            logger.info(f"✓ Saved WoE features only: data/processed/woe_features_only.csv")
    
    if not iv_df.empty:
        # Save IV values
        iv_df.to_csv('../data/processed/iv_values_manual.csv', index=False)
        logger.info(f"✓ Saved IV values: data/processed/iv_values_manual.csv")
        
        # Visualize
        visualize_results(iv_df)
        
        # Print summary
        print("\n" + "="*70)
        print("SUMMARY - MANUAL WOE TRANSFORMATION")
        print("="*70)
        
        total_features = len(iv_df)
        predictive = len(iv_df[iv_df['IV'] >= 0.02])
        strong = len(iv_df[iv_df['IV'] >= 0.3])
        
        print(f"\n📊 Feature Analysis:")
        print(f"   Total features processed: {total_features}")
        print(f"   Predictive features (IV ≥ 0.02): {predictive} ({predictive/total_features:.1%})")
        print(f"   Strong features (IV ≥ 0.3): {strong} ({strong/total_features:.1%})")
        
        print(f"\n🏆 Top 5 Most Predictive Features:")
        for i, (_, row) in enumerate(iv_df.head(5).iterrows(), 1):
            strength = classify_iv_strength(row['IV'])
            print(f"   {i}. {row['Feature']:20} IV={row['IV']:.4f} ({strength})")
        
        print(f"\n📈 Next Steps for Modeling:")
        print(f"   1. Use features with IV ≥ 0.02 for your model")
        print(f"   2. For logistic regression, use the _WOE transformed features")
        print(f"   3. For tree-based models, you can use original or WoE features")
        
        # Save feature selection recommendation
        selected_features = iv_df[iv_df['IV'] >= 0.02]['Feature'].tolist()
        with open('../data/processed/selected_features.txt', 'w') as f:
            f.write("# Features recommended for modeling (IV ≥ 0.02)\n")
            for feat in selected_features:
                f.write(f"{feat}_WOE\n")
        
        logger.info(f"✓ Saved feature selection list: data/processed/selected_features.txt")
    
    else:
        logger.warning("⚠️ No IV values calculated. Check feature preparation.")
    
    return df_woe, iv_df, woe_results

if __name__ == "__main__":
    main()