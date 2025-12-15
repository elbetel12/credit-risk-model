# src/target_engineering_fixed.py
"""
Task 4: Proxy Target Variable Engineering - FIXED for NaN values
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
import os
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_directory(path):
    """Ensure directory exists"""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return True

def save_plot(fig, filename):
    """Save plot with directory check"""
    ensure_directory(filename)
    try:
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Saved: {filename}")
        return True
    except Exception as e:
        logger.error(f"Failed to save {filename}: {e}")
        return False

def load_data():
    """Load data.csv file"""
    logger.info("Loading data...")
    
    paths = [
        'data/raw/data.csv',
        '../data/raw/data.csv',
        'data.csv',
        '../data.csv'
    ]
    
    for path in paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                logger.info(f"✓ Loaded {path}: {df.shape[0]:,} rows, {df.shape[1]} columns")
                return df
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
    
    logger.error("❌ Could not find data.csv")
    return None

def calculate_rfm(df):
    """Calculate RFM metrics with NaN handling"""
    logger.info("\nCalculating RFM metrics...")
    
    # Check for required columns
    required = ['CustomerId', 'Amount', 'TransactionStartTime']
    missing = [col for col in required if col not in df.columns]
    if missing:
        logger.error(f"Missing columns: {missing}")
        return None
    
    # Convert to datetime, handle errors
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
    
    # Check for NaN values
    nan_dates = df['TransactionStartTime'].isna().sum()
    if nan_dates > 0:
        logger.warning(f"⚠️  {nan_dates} rows have invalid dates. These will be excluded.")
        # Drop rows with invalid dates
        df = df.dropna(subset=['TransactionStartTime'])
    
    # Set snapshot date
    snapshot_date = df['TransactionStartTime'].max()
    logger.info(f"Snapshot date: {snapshot_date}")
    
    # Calculate RFM per customer
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'Amount': ['count', 'sum']
    }).reset_index()
    
    # Flatten column names
    rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
    
    logger.info(f"✓ Calculated RFM for {len(rfm):,} customers")
    
    # Check for NaN in RFM
    nan_count = rfm.isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"⚠️  RFM data has {nan_count} NaN values. These will be handled.")
    
    return rfm

def clean_rfm_data(rfm):
    """Clean RFM data by handling NaN and outliers"""
    logger.info("\nCleaning RFM data...")
    
    rfm_clean = rfm.copy()
    
    # Check for NaN values
    nan_before = rfm_clean.isna().sum().sum()
    if nan_before > 0:
        logger.info(f"  Found {nan_before} NaN values")
        
        # Handle NaN in each column
        for col in ['Recency', 'Frequency', 'Monetary']:
            if rfm_clean[col].isna().any():
                # Fill NaN with median
                median_val = rfm_clean[col].median()
                rfm_clean[col] = rfm_clean[col].fillna(median_val)
                logger.info(f"    Filled NaN in {col} with median: {median_val:.2f}")
    
    # Handle zero or negative values for log transformation
    # Frequency should be at least 1
    rfm_clean['Frequency'] = rfm_clean['Frequency'].clip(lower=1)
    
    # Handle negative monetary values (credits/refunds)
    # Use absolute value for clustering
    rfm_clean['Monetary_abs'] = np.abs(rfm_clean['Monetary'])
    
    # Handle extreme outliers
    for col in ['Recency', 'Frequency', 'Monetary_abs']:
        # Cap at 99th percentile
        upper_limit = rfm_clean[col].quantile(0.99)
        if upper_limit > rfm_clean[col].median() * 10:  # Only cap if extreme
            rfm_clean[col] = np.where(rfm_clean[col] > upper_limit, upper_limit, rfm_clean[col])
            logger.info(f"    Capped outliers in {col} at {upper_limit:.2f}")
    
    logger.info("✓ Data cleaning complete")
    return rfm_clean

def visualize_rfm(rfm):
    """Create RFM visualizations"""
    logger.info("\nCreating RFM visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Recency distribution
    axes[0, 0].hist(rfm['Recency'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Recency (days)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Recency Distribution')
    
    # Plot 2: Frequency distribution
    axes[0, 1].hist(rfm['Frequency'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Frequency')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Frequency Distribution')
    
    # Plot 3: Monetary distribution (log scale)
    axes[1, 0].hist(np.log1p(rfm['Monetary_abs']), bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Log(Absolute Monetary + 1)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Monetary Distribution (Log scale)')
    
    # Plot 4: RFM scatter
    scatter = axes[1, 1].scatter(rfm['Recency'], rfm['Frequency'], 
                                 c=np.log1p(rfm['Monetary_abs']), 
                                 alpha=0.6, s=20, cmap='viridis')
    axes[1, 1].set_xlabel('Recency')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Recency vs Frequency (Color = Monetary)')
    plt.colorbar(scatter, ax=axes[1, 1], label='Log(Monetary)')
    
    plt.tight_layout()
    save_plot(fig, 'data/processed/rfm_distributions.png')
    plt.show()
    
    return fig

def create_clusters(rfm_clean, n_clusters=3):
    """Create clusters using K-means with NaN handling"""
    logger.info(f"\nCreating {n_clusters} clusters...")
    
    # Prepare features
    features = rfm_clean[['Recency', 'Frequency', 'Monetary_abs']].copy()
    
    # Apply log transformation (handle any remaining zeros)
    features['Frequency_log'] = np.log1p(features['Frequency'])
    features['Monetary_log'] = np.log1p(features['Monetary_abs'])
    
    # Final check for NaN
    nan_count = features.isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"  Still found {nan_count} NaN values after cleaning")
        # Fill any remaining NaN with column mean
        features = features.fillna(features.mean())
    
    # Select features for clustering
    X = features[['Recency', 'Frequency_log', 'Monetary_log']].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Verify no NaN
    if np.isnan(X_scaled).any():
        logger.error("❌ Still have NaN after scaling. Replacing with 0.")
        X_scaled = np.nan_to_num(X_scaled)
    
    # Apply K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add clusters to RFM data
    rfm_clustered = rfm_clean.copy()
    rfm_clustered['Cluster'] = clusters
    
    # Show distribution
    logger.info("\nCluster Distribution:")
    for cluster_id in range(n_clusters):
        count = (clusters == cluster_id).sum()
        percentage = (count / len(clusters)) * 100
        logger.info(f"  Cluster {cluster_id}: {count:,} customers ({percentage:.1f}%)")
    
    return rfm_clustered, kmeans

def identify_high_risk_cluster(rfm_clustered):
    """Identify high-risk cluster"""
    logger.info("\nIdentifying high-risk cluster...")
    
    # Calculate averages
    cluster_stats = rfm_clustered.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary_abs': 'mean'
    })
    
    logger.info("\nCluster Averages:")
    print(cluster_stats)
    
    # Normalize for comparison (0-1 scale)
    normalized = cluster_stats.copy()
    for col in ['Recency', 'Frequency', 'Monetary_abs']:
        col_min = normalized[col].min()
        col_max = normalized[col].max()
        if col_max > col_min:
            normalized[f'{col}_norm'] = (normalized[col] - col_min) / (col_max - col_min)
    
    # Risk calculation (higher = more risky)
    # High risk = High Recency + Low Frequency + Low Monetary
    normalized['Risk_Score'] = (
        normalized['Recency_norm'] + 
        (1 - normalized['Frequency_norm']) + 
        (1 - normalized['Monetary_abs_norm'])
    ) / 3
    
    # Identify high-risk cluster
    high_risk_cluster = normalized['Risk_Score'].idxmax()
    
    logger.info("\nRisk Analysis:")
    for cluster_id in normalized.index:
        risk_level = "HIGH-RISK" if cluster_id == high_risk_cluster else "Low/Medium Risk"
        logger.info(f"  Cluster {cluster_id}: Score={normalized.loc[cluster_id, 'Risk_Score']:.3f} ({risk_level})")
    
    logger.info(f"\n✅ Identified Cluster {high_risk_cluster} as HIGH-RISK")
    
    return high_risk_cluster, cluster_stats

def create_target_variable(rfm_clustered, high_risk_cluster):
    """Create binary target variable"""
    logger.info("\nCreating target variable...")
    
    rfm_clustered['is_high_risk'] = (rfm_clustered['Cluster'] == high_risk_cluster).astype(int)
    
    # Distribution
    risk_counts = rfm_clustered['is_high_risk'].value_counts()
    risk_percentage = rfm_clustered['is_high_risk'].value_counts(normalize=True) * 100
    
    logger.info("\nTarget Distribution:")
    logger.info(f"  High Risk (1): {risk_counts.get(1, 0):,} ({risk_percentage.get(1, 0):.1f}%)")
    logger.info(f"  Low/Medium Risk (0): {risk_counts.get(0, 0):,} ({risk_percentage.get(0, 0):.1f}%)")
    
    target_df = rfm_clustered[['CustomerId', 'is_high_risk']]
    
    return target_df, rfm_clustered

def visualize_clusters(rfm_clustered, high_risk_cluster):
    """Visualize clusters"""
    logger.info("\nVisualizing clusters...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Colors: red for high-risk, blue for others
    colors = ['red' if c == high_risk_cluster else 'blue' for c in rfm_clustered['Cluster']]
    
    # Plot 1: Recency vs Frequency
    axes[0, 0].scatter(rfm_clustered['Recency'], rfm_clustered['Frequency'], 
                       c=colors, alpha=0.6, s=30)
    axes[0, 0].set_xlabel('Recency')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Recency vs Frequency')
    axes[0, 0].legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='High Risk'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Low/Medium Risk')
    ])
    
    # Plot 2: Cluster distribution
    cluster_counts = rfm_clustered['Cluster'].value_counts().sort_index()
    cluster_colors = ['red' if i == high_risk_cluster else 'blue' for i in cluster_counts.index]
    axes[0, 1].bar(cluster_counts.index, cluster_counts.values, color=cluster_colors)
    axes[0, 1].set_xlabel('Cluster')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Cluster Distribution')
    
    # Plot 3: Risk distribution pie chart
    risk_counts = rfm_clustered['is_high_risk'].value_counts()
    axes[1, 0].pie(risk_counts.values, labels=['Low/Medium Risk', 'High Risk'],
                   autopct='%1.1f%%', colors=['blue', 'red'], startangle=90)
    axes[1, 0].set_title('Risk Distribution')
    
    # Plot 4: RFM radar (simplified bar chart)
    cluster_means = rfm_clustered.groupby('Cluster')[['Recency', 'Frequency', 'Monetary_abs']].mean()
    
    # Normalize
    for col in cluster_means.columns:
        col_min = cluster_means[col].min()
        col_max = cluster_means[col].max()
        if col_max > col_min:
            cluster_means[f'{col}_norm'] = (cluster_means[col] - col_min) / (col_max - col_min)
    
    x = np.arange(3)
    width = 0.25
    
    for i, cluster_id in enumerate(cluster_means.index):
        offset = (i - 1) * width
        values = [
            cluster_means.loc[cluster_id, 'Recency_norm'],
            cluster_means.loc[cluster_id, 'Frequency_norm'],
            cluster_means.loc[cluster_id, 'Monetary_abs_norm']
        ]
        color = 'red' if cluster_id == high_risk_cluster else 'blue'
        axes[1, 1].bar(x + offset, values, width, label=f'Cluster {cluster_id}', color=color)
    
    axes[1, 1].set_xlabel('RFM Metric')
    axes[1, 1].set_ylabel('Normalized Value')
    axes[1, 1].set_title('Cluster Profiles')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(['Recency', 'Frequency', 'Monetary'])
    axes[1, 1].legend()
    
    plt.tight_layout()
    save_plot(fig, 'data/processed/cluster_visualization.png')
    plt.show()
    
    return fig

def save_results(rfm_clustered, target_df, cluster_stats):
    """Save all results"""
    logger.info("\nSaving results...")
    
    ensure_directory('data/processed/')
    
    # Save RFM with clusters
    rfm_clustered.to_csv('data/processed/rfm_with_clusters.csv', index=False)
    logger.info("✓ Saved: data/processed/rfm_with_clusters.csv")
    
    # Save target
    target_df.to_csv('data/processed/target_variable.csv', index=False)
    logger.info("✓ Saved: data/processed/target_variable.csv")
    
    # Save stats
    cluster_stats.to_csv('data/processed/cluster_statistics.csv')
    logger.info("✓ Saved: data/processed/cluster_statistics.csv")
    
    # Try to merge with features
    try:
        features = pd.read_csv('data/processed/customer_features.csv')
        modeling_data = features.merge(target_df, on='CustomerId', how='left')
        modeling_data['is_high_risk'] = modeling_data['is_high_risk'].fillna(0).astype(int)
        modeling_data.to_csv('data/processed/modeling_dataset.csv', index=False)
        logger.info("✓ Saved: data/processed/modeling_dataset.csv")
    except:
        logger.info("ℹ️  Features not found. Target saved separately.")

def main():
    """Main function"""
    print("\n" + "="*70)
    print("TASK 4: PROXY TARGET VARIABLE ENGINEERING")
    print("="*70)
    
    try:
        # Step 1: Load data
        df = load_data()
        if df is None:
            return
        
        # Step 2: Calculate RFM
        rfm = calculate_rfm(df)
        if rfm is None:
            return
        
        # Step 3: Clean data (handle NaN)
        rfm_clean = clean_rfm_data(rfm)
        
        # Step 4: Visualize
        visualize_rfm(rfm_clean)
        
        # Step 5: Create clusters
        rfm_clustered, kmeans = create_clusters(rfm_clean, n_clusters=3)
        
        # Step 6: Identify high-risk cluster
        high_risk_cluster, cluster_stats = identify_high_risk_cluster(rfm_clustered)
        
        # Step 7: Create target
        target_df, rfm_with_target = create_target_variable(rfm_clustered, high_risk_cluster)
        
        # Step 8: Visualize clusters
        visualize_clusters(rfm_with_target, high_risk_cluster)
        
        # Step 9: Save results
        save_results(rfm_with_target, target_df, cluster_stats)
        
        # Summary
        print("\n" + "="*70)
        print("✅ TASK 4 COMPLETE!")
        print("="*70)
        
        high_risk_count = target_df['is_high_risk'].sum()
        total = len(target_df)
        high_risk_pct = (high_risk_count / total) * 100
        
        print(f"\n📊 Results:")
        print(f"  Total customers: {total:,}")
        print(f"  High-risk: {high_risk_count:,} ({high_risk_pct:.1f}%)")
        print(f"  Low/Medium-risk: {total - high_risk_count:,} ({100 - high_risk_pct:.1f}%)")
        
        print(f"\n💾 Files created:")
        for file in ['rfm_with_clusters.csv', 'target_variable.csv', 
                    'cluster_statistics.csv', 'modeling_dataset.csv',
                    'rfm_distributions.png', 'cluster_visualization.png']:
            if os.path.exists(f'data/processed/{file}'):
                print(f"  • {file}")
        
        print(f"\n🚀 Ready for Task 5: Model Training")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()