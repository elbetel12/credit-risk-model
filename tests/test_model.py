import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def trained_model():
    """Create a simple trained model for testing"""
    X = np.array([
        [100, 10, 5, 30],
        [50, 2, 1, 10],
        [500, 20, 10, 100],
        [1000, 50, 20, 365]
    ])
    y = np.array([0, 1, 0, 1])
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model

def test_model_predict_proba_ranges(trained_model):
    """Test that predict_proba returns values between 0 and 1"""
    X_test = np.random.randn(10, 4)
    probs = trained_model.predict_proba(X_test)
    
    assert np.all(probs >= 0)
    assert np.all(probs <= 1)

def test_prediction_sum_to_one(trained_model):
    """Test that prediction sums to 1 for binary classification"""
    X_test = np.random.randn(10, 4)
    probs = trained_model.predict_proba(X_test)
    
    # Sum of probabilities for each sample should be 1
    sums = np.sum(probs, axis=1)
    assert np.allclose(sums, 1.0)

def test_model_batch_vs_single(trained_model):
    """Test model handles single sample vs. batch predictions"""
    X_batch = np.array([
        [100, 10, 5, 30],
        [500, 20, 10, 100]
    ])
    
    # Batch prediction
    preds_batch = trained_model.predict(X_batch)
    assert len(preds_batch) == 2
    
    # Single prediction
    X_single = X_batch[0].reshape(1, -1)
    pred_single = trained_model.predict(X_single)
    assert len(pred_single) == 1
    assert pred_single[0] == preds_batch[0]

def test_model_invalid_input_shape(trained_model):
    """Test model raises appropriate errors for invalid input shapes"""
    # Model expects 4 features
    X_invalid = np.random.randn(5, 3) 
    
    with pytest.raises(ValueError):
        trained_model.predict(X_invalid)
