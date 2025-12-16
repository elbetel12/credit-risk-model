import pytest
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def test_model_loading(tmp_path):
    """Test that we can create and load a model"""
    # Create a simple model
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Save and load
    model_path = tmp_path / "test_model.pkl"
    joblib.dump(model, model_path)
    
    loaded_model = joblib.load(model_path)
    
    # Test predictions match
    test_X = np.random.randn(5, 10)
    pred_original = model.predict(test_X)
    pred_loaded = loaded_model.predict(test_X)
    
    assert np.array_equal(pred_original, pred_loaded)

def test_scaler():
    """Test scaler functionality"""
    scaler = StandardScaler()
    X = np.random.randn(100, 5)
    scaler.fit(X)
    
    X_transformed = scaler.transform(X)
    
    # Check mean is ~0 and std is ~1
    assert np.allclose(X_transformed.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(X_transformed.std(axis=0), 1, atol=1e-10)