print('Start Loading SHAP...')
import shap
print(f'SHAP Version: {shap.__version__}')
import joblib
print('Loading Model...')
model = joblib.load('models/best_model.pkl')
print('Model Loaded.')
