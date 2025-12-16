# src/train_clean.py
"""
Task 5: Clean Model Training Script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix,
                           roc_curve, auc)
import logging
import os
import warnings
import joblib
warnings.filterwarnings('ignore')

# Import ML models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline

# MLflow for experiment tracking
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup():
    """Setup environment"""
    print("\n" + "="*70)
    print("TASK 5: MODEL TRAINING AND TRACKING")
    print("="*70)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Setup MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Credit_Risk_Modeling")
    
    # Create MLflow experiment if it doesn't exist
    experiment = mlflow.get_experiment_by_name("Credit_Risk_Modeling")
    if experiment is None:
        mlflow.create_experiment("Credit_Risk_Modeling")
    
    logger.info("MLflow tracking URI: sqlite:///mlflow.db")
    logger.info("MLflow experiment: Credit_Risk_Modeling")

def load_and_validate_data():
    """Load and validate data"""
    logger.info("📂 Loading data...")
    
    features_path = 'data/processed/customer_features.csv'
    target_path = 'data/processed/target_variable.csv'
    
    # Check files exist
    if not os.path.exists(features_path):
        logger.error(f"❌ Features file not found: {features_path}")
        return None
    
    if not os.path.exists(target_path):
        logger.error(f"❌ Target file not found: {target_path}")
        return None
    
    try:
        # Load data
        features = pd.read_csv(features_path)
        target = pd.read_csv(target_path)
        
        logger.info(f"✓ Features shape: {features.shape}")
        logger.info(f"✓ Target shape: {target.shape}")
        
        # FIX: Convert date columns properly
        date_columns = features.select_dtypes(include=['object']).columns.tolist()
        date_columns = [col for col in date_columns if any(date_term in col.lower() for date_term in ['date', 'time', 'transaction'])]
        
        if date_columns:
            logger.info(f"Converting date columns to numeric: {date_columns}")
            for col in date_columns:
                try:
                    # Clean the date strings
                    features[col] = features[col].astype(str)
                    
                    # Remove extra leading zeros (fix for '002019-02-02' format)
                    features[col] = features[col].str.lstrip('0')
                    
                    # Convert to datetime
                    features[col] = pd.to_datetime(features[col], errors='coerce')
                    
                    # Convert to numeric (days since epoch)
                    features[col] = (features[col] - pd.Timestamp("1970-01-01")).dt.days
                    
                    # Fill NaN with median
                    median_val = features[col].median()
                    features[col] = features[col].fillna(median_val)
                    
                    logger.info(f"  {col}: converted to days since epoch")
                    
                except Exception as e:
                    logger.warning(f"  Could not convert {col}: {e}")
                    # Drop if conversion fails
                    if col in features.columns:
                        features = features.drop(col, axis=1)
        
        # Merge
        if 'CustomerId' not in features.columns or 'CustomerId' not in target.columns:
            logger.error("❌ Missing CustomerId column")
            return None
        
        data = features.merge(target, on='CustomerId', how='inner')
        logger.info(f"✓ Merged data shape: {data.shape}")
        
        # Show target distribution
        counts = data['is_high_risk'].value_counts()
        percentages = data['is_high_risk'].value_counts(normalize=True) * 100
        
        logger.info(f"\n📊 Target Distribution:")
        logger.info(f"  High Risk (1): {counts.get(1, 0):,} samples ({percentages.get(1, 0):.1f}%)")
        logger.info(f"  Low Risk (0): {counts.get(0, 0):,} samples ({percentages.get(0, 0):.1f}%)")
        
        return data
        
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        return None

def create_engineered_features(data):
    """Create enhanced features including date-based features"""
    logger.info("\n🔧 Creating engineered features...")
    
    # Make a copy to avoid modifying original
    df = data.copy()
    
    # Ensure we have the required base columns
    required_cols = ['total_amount', 'transaction_count', 'first_transaction', 'last_transaction']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.warning(f"⚠️ Missing columns for feature engineering: {missing_cols}")
        logger.info("  Using available columns only")
    
    # Create basic monetary features (if they exist)
    if 'total_amount' in df.columns:
        # Already have total_amount
        
        # Create average amount per transaction
        if 'transaction_count' in df.columns:
            df['avg_amount'] = df['total_amount'] / df['transaction_count'].replace(0, 1)
            logger.info("  Created: avg_amount")
        
        # Create transaction frequency (if we have time data)
        if 'first_transaction' in df.columns and 'last_transaction' in df.columns:
            # Calculate days between first and last transaction
            df['transaction_period_days'] = df['last_transaction'] - df['first_transaction']
            df['transaction_period_days'] = df['transaction_period_days'].clip(lower=1)  # Avoid division by zero
            
            # Create meaningful date-based features
            df['days_since_first_transaction'] = df['first_transaction']
            df['days_since_last_transaction'] = df['last_transaction']
            
            # Calculate transaction frequency (transactions per day)
            df['transaction_frequency'] = df['transaction_count'] / df['transaction_period_days']
            logger.info("  Created: transaction_frequency")
            
            # Customer tenure in weeks
            df['customer_tenure_weeks'] = df['days_since_first_transaction'] / 7
            logger.info("  Created: customer_tenure_weeks")
            
            # Recent activity indicator (active in last 30 days)
            df['active_recently'] = (df['days_since_last_transaction'] <= 30).astype(int)
            logger.info("  Created: active_recently")
            
            # Inactivity ratio (days between transactions)
            df['avg_days_between_transactions'] = df['transaction_period_days'] / df['transaction_count'].replace(0, 1)
            logger.info("  Created: avg_days_between_transactions")
    
    # Create amount-based features
    if 'total_amount' in df.columns:
        # Log transform for skewed amounts
        df['log_total_amount'] = np.log1p(df['total_amount'])
        logger.info("  Created: log_total_amount")
        
        # Amount categories
        df['amount_category'] = pd.cut(df['total_amount'], 
                                      bins=[-np.inf, 100, 500, 2000, np.inf],
                                      labels=['very_low', 'low', 'medium', 'high'])
        
        # Convert to numerical (one-hot would be better but keeping simple)
        df['amount_category_code'] = df['amount_category'].cat.codes
        logger.info("  Created: amount_category_code")
    
    # Create transaction count features
    if 'transaction_count' in df.columns:
        # Log transform
        df['log_transaction_count'] = np.log1p(df['transaction_count'])
        logger.info("  Created: log_transaction_count")
        
        # Transaction count categories
        df['tx_count_category'] = pd.cut(df['transaction_count'],
                                        bins=[-np.inf, 5, 20, 50, np.inf],
                                        labels=['few', 'moderate', 'many', 'very_many'])
        df['tx_count_category_code'] = df['tx_count_category'].cat.codes
        logger.info("  Created: tx_count_category_code")
    
    # Create interaction features
    if 'total_amount' in df.columns and 'transaction_count' in df.columns:
        df['avg_txn_size'] = df['total_amount'] / df['transaction_count'].replace(0, 1)
        logger.info("  Created: avg_txn_size")
        
        # Monetary value per day if we have time data
        if 'transaction_period_days' in df.columns:
            df['monetary_per_day'] = df['total_amount'] / df['transaction_period_days']
            logger.info("  Created: monetary_per_day")
    
    # Drop categorical columns that we encoded
    cols_to_drop = ['amount_category', 'tx_count_category', 'transaction_period_days']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    logger.info(f"  Original features: {list(data.columns)}")
    logger.info(f"  Engineered features added. New shape: {df.shape}")
    
    return df

def prepare_data(data):
    """Prepare data for training with feature selection"""
    logger.info("\n🔧 Preparing data for training...")
    
    # Create engineered features
    data_engineered = create_engineered_features(data)
    
    # Select final features for modeling
    # Prioritize these features based on domain knowledge
    preferred_features = [
        # Core monetary features
        'total_amount', 'avg_amount', 'log_total_amount',
        
        # Transaction features
        'transaction_count', 'log_transaction_count', 'avg_txn_size',
        
        # Date-based features (CRITICAL for credit risk)
        'days_since_last_transaction', 'days_since_first_transaction',
        'transaction_frequency', 'avg_days_between_transactions',
        'active_recently', 'customer_tenure_weeks',
        
        # Derived features
        'monetary_per_day', 'amount_category_code', 'tx_count_category_code'
    ]
    
    # Filter to only include features that exist in our data
    available_features = [f for f in preferred_features if f in data_engineered.columns]
    
    # Add CustomerId and target for splitting
    if 'CustomerId' not in data_engineered.columns:
        logger.error("❌ CustomerId column missing")
        return None
    
    # Separate features and target
    X = data_engineered[available_features].copy()
    y = data_engineered['is_high_risk']
    
    logger.info(f"  Selected {len(available_features)} features for modeling:")
    for i, feat in enumerate(available_features, 1):
        logger.info(f"    {i:2d}. {feat}")
    
    # Handle any remaining non-numeric columns
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        logger.warning(f"⚠️ Non-numeric columns found: {non_numeric_cols}")
        for col in non_numeric_cols:
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                logger.info(f"  Converted {col} to numeric")
            except:
                X = X.drop(col, axis=1)
                logger.info(f"  Dropped {col}")
    
    # Handle missing values
    nan_count = X.isnull().sum().sum()
    if nan_count > 0:
        logger.info(f"  Found {nan_count} NaN values in features")
        
        # Check which columns have missing values
        nan_columns = X.columns[X.isnull().any()].tolist()
        logger.info(f"  Columns with NaN: {nan_columns}")
        
        # Fill with median (better for skewed data)
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
                logger.info(f"  Filled NaN in {col} with median: {median_val:.4f}")
    
    # Save feature names for later use (CRITICAL for API)
    feature_names = X.columns.tolist()
    
    # Log feature statistics
    logger.info("\n📊 Feature Statistics:")
    for feat in feature_names[:10]:  # Show first 10
        logger.info(f"  {feat:30s} mean: {X[feat].mean():8.2f} std: {X[feat].std():8.2f}")
    
    if len(feature_names) > 10:
        logger.info(f"  ... and {len(feature_names) - 10} more features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    logger.info(f"\n📊 Data Split:")
    logger.info(f"  Training: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    logger.info(f"  Testing:  {X_test.shape[0]:,} samples, {X_test.shape[1]} features")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for MLflow signature
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    # Save scaler and feature names
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Save feature names for API to use
    feature_info = {
        'feature_names': feature_names,
        'feature_importances': {}  # Will be filled after training
    }
    joblib.dump(feature_info, 'models/feature_info.pkl')
    
    logger.info("✓ Scaler saved: models/scaler.pkl")
    logger.info("✓ Feature info saved: models/feature_info.pkl")
    
    # Log feature names to file for reference
    with open('data/processed/model_features.txt', 'w') as f:
        f.write("Features used in model training:\n")
        f.write("=" * 50 + "\n")
        for i, feat in enumerate(feature_names, 1):
            f.write(f"{i:3d}. {feat}\n")
    
    logger.info("✓ Feature list saved: data/processed/model_features.txt")
    
    return X_train_scaled_df, X_test_scaled_df, y_train, y_test, feature_names

def train_model_with_mlflow(model_name, model, X_train, y_train, X_test, y_test, feature_names, run_name="model_training"):
    """Train a single model with MLflow tracking"""
    logger.info(f"\n🎯 Training {model_name} with MLflow...")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"{run_name}_{model_name}"):
        try:
            # Log parameters
            if model_name == 'Logistic Regression':
                mlflow.log_param("model_type", "LogisticRegression")
                mlflow.log_param("max_iter", 1000)
                mlflow.log_param("C", 1.0)
            elif model_name == 'Random Forest':
                mlflow.log_param("model_type", "RandomForestClassifier")
                mlflow.log_param("n_estimators", 100)
                mlflow.log_param("random_state", 42)
                mlflow.log_param("max_depth", "None")
            elif model_name == 'Gradient Boosting':
                mlflow.log_param("model_type", "GradientBoostingClassifier")
                mlflow.log_param("n_estimators", 100)
                mlflow.log_param("learning_rate", 0.1)
                mlflow.log_param("max_depth", 3)
            
            # Log feature information
            mlflow.log_param("n_features", len(feature_names))
            mlflow.log_param("features", ", ".join(feature_names[:10]))  # Log first 10
            
            # Create a pipeline with imputer for models that don't handle NaN
            if model_name in ['Logistic Regression', 'Gradient Boosting']:
                pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('model', model)
                ])
                pipeline.fit(X_train, y_train)
                trained_model = pipeline
            else:
                # Random Forest can handle some NaN when using median imputation internally
                model.fit(X_train, y_train)
                trained_model = model
            
            # Make predictions
            if model_name in ['Logistic Regression', 'Gradient Boosting']:
                y_pred = trained_model.predict(X_test)
                y_pred_proba = trained_model.predict_proba(X_test)[:, 1]
            else:
                y_pred = trained_model.predict(X_test)
                y_pred_proba = trained_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Calculate and log feature importance for tree-based models
            if model_name in ['Random Forest', 'Gradient Boosting']:
                if hasattr(trained_model, 'feature_importances_'):
                    importances = trained_model.feature_importances_
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importances
                    }).sort_values('importance', ascending=False)
                    
                    # Log top 10 features
                    top_features = importance_df.head(10)
                    logger.info(f"\n🔝 Top 10 Important Features for {model_name}:")
                    for idx, row in top_features.iterrows():
                        logger.info(f"  {row['feature']:30s}: {row['importance']:.4f}")
                    
                    # Save feature importance
                    importance_path = f"data/processed/feature_importance_{model_name.lower().replace(' ', '_')}.csv"
                    importance_df.to_csv(importance_path, index=False)
                    mlflow.log_artifact(importance_path)
            
            # Log model to MLflow with signature
            signature = infer_signature(X_train, y_pred)
            mlflow.sklearn.log_model(
                trained_model,
                artifact_path=f"{model_name.lower().replace(' ', '_')}_model",
                signature=signature,
                registered_model_name=f"credit_risk_{model_name.lower().replace(' ', '_')}"
            )
            
            # Log confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, 
                                index=['Actual Low Risk', 'Actual High Risk'],
                                columns=['Predicted Low Risk', 'Predicted High Risk'])
            
            cm_path = f"data/processed/confusion_matrix_{model_name.lower().replace(' ', '_')}.csv"
            cm_df.to_csv(cm_path)
            mlflow.log_artifact(cm_path)
            
            logger.info(f"  ✓ {model_name} trained and logged to MLflow")
            logger.info(f"    ROC-AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"    Accuracy: {metrics['accuracy']:.4f}")
            
            return {
                'model': trained_model,
                'metrics': metrics,
                'name': model_name,
                'feature_names': feature_names
            }
            
        except Exception as e:
            logger.error(f"  ✗ Error training {model_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

def train_all_models_with_mlflow(X_train, X_test, y_train, y_test, feature_names):
    """Train all models with MLflow tracking"""
    logger.info("\n" + "="*60)
    logger.info("TRAINING MODELS WITH MLFLOW")
    logger.info("="*60)
    
    # Define models
    models_to_train = [
        {
            'name': 'Logistic Regression',
            'model': LogisticRegression(random_state=42, max_iter=1000, C=1.0, class_weight='balanced'),
        },
        {
            'name': 'Random Forest',
            'model': RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'),
        },
        {
            'name': 'Gradient Boosting',
            'model': GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1),
        }
    ]
    
    results = {}
    
    for model_config in models_to_train:
        result = train_model_with_mlflow(
            model_config['name'],
            model_config['model'],
            X_train, y_train, X_test, y_test,
            feature_names
        )
        
        if result:
            results[model_config['name']] = result
    
    return results

def compare_models(results):
    """Compare and select best model"""
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON")
    logger.info("="*60)
    
    if not results:
        logger.error("❌ No models trained successfully")
        return None, None
    
    # Create comparison table
    comparison_data = []
    for model_name, result in results.items():
        row = {
            'Model': model_name,
            'ROC-AUC': result['metrics']['roc_auc'],
            'Accuracy': result['metrics']['accuracy'],
            'Precision': result['metrics']['precision'],
            'Recall': result['metrics']['recall'],
            'F1-Score': result['metrics']['f1'],
            'Num_Features': len(result['feature_names'])
        }
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*60)
    print(comparison_df.round(4).to_string(index=False))
    
    # Select best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_result = results[best_model_name]
    
    logger.info(f"\n🏆 BEST MODEL: {best_model_name}")
    logger.info(f"   ROC-AUC: {best_result['metrics']['roc_auc']:.4f}")
    logger.info(f"   Features used: {len(best_result['feature_names'])}")
    
    # Save best model locally
    try:
        joblib.dump(best_result['model'], 'models/best_model.pkl')
        logger.info("✓ Best model saved: models/best_model.pkl")
        
        # Update feature info with importances if available
        feature_info = joblib.load('models/feature_info.pkl')
        if hasattr(best_result['model'], 'feature_importances_'):
            feature_info['feature_importances'] = best_result['model'].feature_importances_.tolist()
        joblib.dump(feature_info, 'models/feature_info.pkl')
        
        # Log best model metrics to MLflow
        with mlflow.start_run(run_name="best_model_summary"):
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_param("best_model_features", len(best_result['feature_names']))
            
            for metric_name, metric_value in best_result['metrics'].items():
                mlflow.log_metric(f"best_{metric_name}", metric_value)
            
            # Log comparison table
            comparison_path = "data/processed/model_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            mlflow.log_artifact(comparison_path)
            
            # Log feature list
            features_path = "data/processed/best_model_features.txt"
            with open(features_path, 'w') as f:
                f.write(f"Best Model: {best_model_name}\n")
                f.write(f"ROC-AUC: {best_result['metrics']['roc_auc']:.4f}\n")
                f.write("\nFeatures:\n")
                f.write("="*50 + "\n")
                for i, feat in enumerate(best_result['feature_names'], 1):
                    f.write(f"{i:3d}. {feat}\n")
            mlflow.log_artifact(features_path)
            
    except Exception as e:
        logger.error(f"❌ Failed to save model: {e}")
    
    return best_result, comparison_df

def create_simple_visualization(results, X_test, y_test, comparison_df):
    """Create simple visualization"""
    logger.info("\n📈 Creating visualization...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Model comparison (top-left)
        models = comparison_df['Model'].values
        roc_scores = comparison_df['ROC-AUC'].values
        
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        bars = axes[0, 0].bar(models, roc_scores, color=colors, edgecolor='black')
        axes[0, 0].set_ylabel('ROC-AUC Score', fontsize=12)
        axes[0, 0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, score in zip(bars, roc_scores):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                          f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: ROC curve for best model (top-right)
        best_model_name = comparison_df.iloc[0]['Model']
        best_model = results[best_model_name]['model']
        
        # Get predictions
        if hasattr(best_model, 'predict_proba'):
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'{best_model_name} (AUC = {roc_auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        axes[0, 1].set_xlabel('False Positive Rate', fontsize=12)
        axes[0, 1].set_ylabel('True Positive Rate', fontsize=12)
        axes[0, 1].set_title(f'ROC Curve - {best_model_name}', fontsize=14, fontweight='bold')
        axes[0, 1].legend(loc='lower right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Feature importance for best model if available (bottom-left)
        if best_model_name in ['Random Forest', 'Gradient Boosting']:
            if hasattr(best_model, 'feature_importances_'):
                feature_names = results[best_model_name]['feature_names']
                importances = best_model.feature_importances_
                
                # Get top 10 features
                indices = np.argsort(importances)[-10:][::-1]
                top_features = [feature_names[i] for i in indices]
                top_importances = importances[indices]
                
                colors_bar = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
                axes[1, 0].barh(range(len(top_features)), top_importances, color=colors_bar, edgecolor='black')
                axes[1, 0].set_yticks(range(len(top_features)))
                axes[1, 0].set_yticklabels(top_features)
                axes[1, 0].set_xlabel('Importance', fontsize=12)
                axes[1, 0].set_title(f'Top 10 Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
                axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # Plot 4: Metrics comparison (bottom-right)
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x_pos = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics_to_plot):
            values = comparison_df[metric].values
            axes[1, 1].bar(x_pos + i*width, values, width, label=metric, alpha=0.8)
        
        axes[1, 1].set_xlabel('Models', fontsize=12)
        axes[1, 1].set_ylabel('Score', fontsize=12)
        axes[1, 1].set_title('Model Metrics Comparison', fontsize=14, fontweight='bold')
        axes[1, 1].set_xticks(x_pos + width*1.5)
        axes[1, 1].set_xticklabels(models)
        axes[1, 1].legend(loc='upper right')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('data/processed/model_evaluation.png', dpi=150, bbox_inches='tight')
        
        # Log visualization to MLflow
        with mlflow.start_run(run_name="visualization_log"):
            mlflow.log_artifact('data/processed/model_evaluation.png')
        
        logger.info("✓ Visualization saved and logged to MLflow")
        plt.show()
        
    except Exception as e:
        logger.error(f"❌ Error creating visualization: {e}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """Main function"""
    setup()
    
    # Step 1: Load data
    data = load_and_validate_data()
    if data is None:
        return
    
    # Step 2: Prepare data with engineered features
    X_train, X_test, y_train, y_test, feature_names = prepare_data(data)
    
    # Step 3: Train models with MLflow
    results = train_all_models_with_mlflow(X_train, X_test, y_train, y_test, feature_names)
    
    if not results:
        logger.error("❌ No models were trained successfully")
        return
    
    # Step 4: Compare and select best model
    best_result, comparison_df = compare_models(results)
    
    if best_result is None:
        return
    
    # Step 5: Create visualization
    create_simple_visualization(results, X_test, y_test, comparison_df)
    
    # Final summary
    print("\n" + "="*70)
    print("TASK 5 COMPLETE - MLFLOW SUMMARY")
    print("="*70)
    
    print(f"\nResults Summary:")
    print(f"  Best Model: {comparison_df.iloc[0]['Model']}")
    print(f"  ROC-AUC:    {comparison_df.iloc[0]['ROC-AUC']:.4f}")
    print(f"  Accuracy:   {comparison_df.iloc[0]['Accuracy']:.4f}")
    print(f"  F1-Score:   {comparison_df.iloc[0]['F1-Score']:.4f}")
    print(f"  Features:   {comparison_df.iloc[0]['Num_Features']}")
    
    print(f"\nFeature Categories:")
    date_features = [f for f in feature_names if 'day' in f or 'recent' in f or 'tenure' in f]
    amount_features = [f for f in feature_names if 'amount' in f or 'monetary' in f]
    transaction_features = [f for f in feature_names if 'transaction' in f and 'amount' not in f]
    
    print(f"  Date-based:    {len(date_features)} features")
    print(f"  Amount-based:  {len(amount_features)} features")
    print(f"  Transaction:   {len(transaction_features)} features")
    
    print(f"\nMLflow Information:")
    print(f"  Database: sqlite:///mlflow.db")
    print(f"  Experiment: Credit_Risk_Modeling")
    print(f"  Models logged: {len(results)} models")
    
    print(f"\nTo view results:")
    print(f"  1. Start MLflow UI: mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print(f"  2. Open browser: http://localhost:5000")
    
    print(f"\nFor API deployment (Task 6):")
    print(f"  Features expected: {len(feature_names)}")
    print(f"  See: data/processed/model_features.txt for complete list")
    
    print(f"\nLocal files created:")
    print(f"  1. models/best_model.pkl - Best trained model")
    print(f"  2. models/scaler.pkl - Feature scaler")
    print(f"  3. models/feature_info.pkl - Feature metadata")
    print(f"  4. data/processed/model_evaluation.png - Performance chart")
    print(f"  5. data/processed/model_features.txt - Feature list")

if __name__ == "__main__":
    main()