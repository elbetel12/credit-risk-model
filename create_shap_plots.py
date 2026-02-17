import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("🚀 Starting SHAP analysis script...")

# 1. Configuration and Paths
MODEL_PATH = 'models/best_model.pkl'
FEATURES_PATH = 'data/processed/customer_features.csv'
TARGET_PATH = 'data/processed/target_variable.csv'
INFO_PATH = 'models/feature_info.pkl'
SCALER_PATH = 'models/scaler.pkl'

# 2. Check Prerequisites
for p in [MODEL_PATH, FEATURES_PATH, TARGET_PATH, INFO_PATH, SCALER_PATH]:
    if not os.path.exists(p):
        print(f"❌ Missing required file: {p}")
        exit(1)

# 3. Load Model and Feature Info
print("📂 Loading model and metadata...")
rf_model = joblib.load(MODEL_PATH)
feat_info = joblib.load(INFO_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = feat_info.get('feature_names', [])

# 4. Reconstruct Dataset and X_test
print("🔧 Reconstructing test dataset...")
f_df = pd.read_csv(FEATURES_PATH)
t_df = pd.read_csv(TARGET_PATH)
merged = f_df.merge(t_df, on='CustomerId', how='inner')

# Reconstruct missing features expected by the model
print("🛠️ Engineering missing features...")

if f'log_total_amount' not in merged.columns:
    merged['log_total_amount'] = np.log1p(merged['total_amount'].clip(lower=0))
    print("  Added: log_total_amount")

if f'log_transaction_count' not in merged.columns:
    merged['log_transaction_count'] = np.log1p(merged['transaction_count'].clip(lower=0))
    print("  Added: log_transaction_count")

if f'avg_txn_size' not in merged.columns:
    merged['avg_txn_size'] = merged['total_amount'] / merged['transaction_count'].replace(0, 1)
    print("  Added: avg_txn_size")

if f'amount_category_code' not in merged.columns:
    merged['amount_category_code'] = pd.cut(
        merged['total_amount'], 
        bins=[-np.inf, 100, 500, 2000, np.inf], 
        labels=[0, 1, 2, 3]
    ).astype(float)
    print("  Added: amount_category_code")

if f'tx_count_category_code' not in merged.columns:
    merged['tx_count_category_code'] = pd.cut(
        merged['transaction_count'], 
        bins=[-np.inf, 5, 20, 50, np.inf], 
        labels=[0, 1, 2, 3]
    ).astype(float)
    print("  Added: tx_count_category_code")

# Final check for columns
missing = [col for col in feature_names if col not in merged.columns]
if missing:
    print(f"⚠️ Still missing columns: {missing}")
    # Final desperation fallback for missing ones
    for col in missing:
        merged[col] = 0.0

# Select features and split
X = merged[feature_names]
y = merged['is_high_risk']
_, X_test_raw, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply scaling
X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=feature_names)

# 5. SHAP Analysis
print("🧠 Initializing SHAP Explainer...")

# Handle Pipeline objects (SHAP TreeExplainer needs the actual model)
if hasattr(rf_model, 'named_steps') and 'model' in rf_model.named_steps:
    print(f"  Extracting model from pipeline: {type(rf_model.named_steps['model'])}")
    final_model = rf_model.named_steps['model']
else:
    final_model = rf_model

# Select appropriate explainer
# TreeExplainer for Random Forest / Gradient Boosting
try:
    explainer = shap.TreeExplainer(final_model)
except Exception as e:
    print(f"  Note: TreeExplainer failed ({e}), falling back to KernelExplainer...")
    explainer = shap.KernelExplainer(rf_model.predict_proba, X_test.sample(20))

X_sample = X_test.sample(min(200, len(X_test)), random_state=42)

print("⚡ Calculating SHAP values...")
# For Gradient Boosting, we might need to handle the output differently
shap_values = explainer.shap_values(X_sample)

# Handle output format differences
if isinstance(shap_values, list):
    # Binary classification usually returns [values_class0, values_class1]
    shap_values_risk = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    if hasattr(explainer, 'expected_value') and isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1:
        base_value = explainer.expected_value[1]
    else:
        base_value = explainer.expected_value
else:
    # Some models return values directly for the positive class or require simple indexing
    shap_values_risk = shap_values
    base_value = explainer.expected_value

# 6. Generate and Save Plots
print("📊 Generating visualizations...")
plt.style.use('ggplot')

# Summary Plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_risk, X_sample, show=False)
plt.title('SHAP Summary Plot: Feature Impact on Credit Risk', fontsize=16)
plt.tight_layout()
plt.savefig('shap_summary_plot.png', dpi=150, bbox_inches='tight')
print("✅ Saved: shap_summary_plot.png")

# Find probabilities for the original pipeline to ensure consistency
probs = rf_model.predict_proba(X_sample)[:, 1]
high_risk_idx = np.argmax(probs)
low_risk_idx = np.argmin(probs)

# Waterfall High Risk
plt.figure(figsize=(12, 8))
try:
    exp_high = shap.Explanation(
        values=shap_values_risk[high_risk_idx],
        base_values=base_value,
        data=X_sample.iloc[high_risk_idx],
        feature_names=X_sample.columns.tolist()
    )
    shap.waterfall_plot(exp_high, show=False)
except Exception as e:
    print(f"  Note: Using fallback plot for waterfall ({e})")
    shap.plots.bar(shap_values_risk[high_risk_idx], show=False)

plt.title(f'High-Risk Customer Decision (Risk Score: {probs[high_risk_idx]:.1%})')
plt.tight_layout()
plt.savefig('shap_waterfall_highrisk.png', dpi=150, bbox_inches='tight')
print("✅ Saved: shap_waterfall_highrisk.png")

# Waterfall Low Risk
plt.figure(figsize=(12, 8))
try:
    exp_low = shap.Explanation(
        values=shap_values_risk[low_risk_idx],
        base_values=base_value,
        data=X_sample.iloc[low_risk_idx],
        feature_names=X_sample.columns.tolist()
    )
    shap.waterfall_plot(exp_low, show=False)
except Exception as e:
    shap.plots.bar(shap_values_risk[low_risk_idx], show=False)

plt.title(f'Low-Risk Customer Decision (Risk Score: {probs[low_risk_idx]:.1%})')
plt.tight_layout()
plt.savefig('shap_waterfall_lowrisk.png', dpi=150, bbox_inches='tight')
print("✅ Saved: shap_waterfall_lowrisk.png")

# 7. Save SHAP Values and Explainer Metadata
print("💾 Saving SHAP results to shap_values_sample.pkl...")
shap_output = {
    'shap_values': shap_values_risk,
    'X_sample': X_sample,
    'base_value': base_value,
    'feature_names': X_sample.columns.tolist()
}
joblib.dump(shap_output, 'shap_values_sample.pkl')
print("✅ Saved: shap_values_sample.pkl")

print("\n✨ SHAP Analysis Complete! All plots and data have been generated successfully.")
