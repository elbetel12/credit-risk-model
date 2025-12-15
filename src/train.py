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

def prepare_data(data):
    """Prepare data for training"""
    logger.info("\n🔧 Preparing data...")
    
    # Separate features and target
    X = data.drop(['CustomerId', 'is_high_risk'], axis=1, errors='ignore')
    y = data['is_high_risk']
    
    logger.info(f"  Features: {X.shape}")
    
    # Handle any remaining non-numeric columns
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        logger.warning(f"⚠️ Non-numeric columns found: {non_numeric_cols}")
        
        for col in non_numeric_cols:
            try:
                # Try to convert to numeric
                X[col] = pd.to_numeric(X[col], errors='coerce')
                logger.info(f"  Converted {col} to numeric")
            except:
                # Drop if can't convert
                X = X.drop(col, axis=1)
                logger.info(f"  Dropped {col}")
    
    # Check for NaN values
    nan_count = X.isnull().sum().sum()
    if nan_count > 0:
        logger.info(f"  Found {nan_count} NaN values in features")
        # Fill with median (better than mean for skewed data)
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
        logger.info(f"  Filled NaN values with median")
    
    # Save feature names for later
    feature_names = X.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    logger.info(f"\n📊 Data Split:")
    logger.info(f"  Training: {X_train.shape[0]:,} samples")
    logger.info(f"  Testing:  {X_test.shape[0]:,} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for MLflow signature
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    logger.info("✓ Scaler saved: models/scaler.pkl")
    
    return X_train_scaled_df, X_test_scaled_df, y_train, y_test, feature_names

def train_model_with_mlflow(model_name, model, X_train, y_train, X_test, y_test, run_name="model_training"):
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
            elif model_name == 'Gradient Boosting':
                mlflow.log_param("model_type", "GradientBoostingClassifier")
                mlflow.log_param("n_estimators", 100)
                mlflow.log_param("learning_rate", 0.1)
            
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
            
            # Save confusion matrix as CSV
            cm_path = f"data/processed/confusion_matrix_{model_name.lower().replace(' ', '_')}.csv"
            cm_df.to_csv(cm_path)
            mlflow.log_artifact(cm_path)
            
            logger.info(f"  ✓ {model_name} trained and logged to MLflow")
            logger.info(f"    ROC-AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"    Accuracy: {metrics['accuracy']:.4f}")
            
            return {
                'model': trained_model,
                'metrics': metrics,
                'name': model_name
            }
            
        except Exception as e:
            logger.error(f"  ✗ Error training {model_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

def train_all_models_with_mlflow(X_train, X_test, y_train, y_test):
    """Train all models with MLflow tracking"""
    logger.info("\n" + "="*60)
    logger.info("TRAINING MODELS WITH MLFLOW")
    logger.info("="*60)
    
    # Define models
    models_to_train = [
        {
            'name': 'Logistic Regression',
            'model': LogisticRegression(random_state=42, max_iter=1000, C=1.0),
        },
        {
            'name': 'Random Forest',
            'model': RandomForestClassifier(random_state=42, n_estimators=100),
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
            X_train, y_train, X_test, y_test
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
            'F1-Score': result['metrics']['f1']
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
    
    # Save best model locally as well
    try:
        joblib.dump(best_result['model'], 'models/best_model.pkl')
        logger.info("✓ Best model saved locally: models/best_model.pkl")
        
        # Log best model metrics to a new MLflow run
        with mlflow.start_run(run_name="best_model_summary"):
            mlflow.log_param("best_model", best_model_name)
            for metric_name, metric_value in best_result['metrics'].items():
                mlflow.log_metric(f"best_{metric_name}", metric_value)
            
            # Log comparison table
            comparison_path = "data/processed/model_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            mlflow.log_artifact(comparison_path)
            
    except Exception as e:
        logger.error(f"❌ Failed to save model: {e}")
    
    return best_result, comparison_df

def create_simple_visualization(results, X_test, y_test, comparison_df):
    """Create simple visualization"""
    logger.info("\n📈 Creating visualization...")
    
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Model comparison
        models = comparison_df['Model'].values
        roc_scores = comparison_df['ROC-AUC'].values
        
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        bars = axes[0].bar(models, roc_scores, color=colors, edgecolor='black')
        axes[0].set_ylabel('ROC-AUC Score', fontsize=12)
        axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, score in zip(bars, roc_scores):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: ROC curve for best model
        best_model_name = comparison_df.iloc[0]['Model']
        best_model = results[best_model_name]['model']
        
        # Get predictions
        if hasattr(best_model, 'predict_proba'):
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'{best_model_name} (AUC = {roc_auc:.3f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        axes[1].set_xlabel('False Positive Rate', fontsize=12)
        axes[1].set_ylabel('True Positive Rate', fontsize=12)
        axes[1].set_title(f'ROC Curve - {best_model_name}', fontsize=14, fontweight='bold')
        axes[1].legend(loc='lower right')
        axes[1].grid(True, alpha=0.3)
        
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
    
    # Step 2: Prepare data
    X_train, X_test, y_train, y_test, feature_names = prepare_data(data)
    
    # Step 3: Train models with MLflow
    results = train_all_models_with_mlflow(X_train, X_test, y_train, y_test)
    
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
    
    print(f"\nMLflow Information:")
    print(f"  Database: sqlite:///mlflow.db")
    print(f"  Experiment: Credit_Risk_Modeling")
    print(f"  Models logged: {len(results)} models")
    
    print(f"\nTo view results:")
    print(f"  1. Start MLflow UI: mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print(f"  2. Open browser: http://localhost:5000")
    print(f"  3. Click on 'Credit_Risk_Modeling' experiment")
    print(f"  4. View runs and models under each run")
    
    print(f"\nLocal files created:")
    print(f"  1. models/best_model.pkl - Best trained model")
    print(f"  2. models/scaler.pkl - Feature scaler")
    print(f"  3. data/processed/model_evaluation.png - Performance chart")
    print(f"  4. data/processed/model_comparison.csv - Comparison table")

if __name__ == "__main__":
    main()