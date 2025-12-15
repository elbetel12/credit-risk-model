# tests/test_training.py
"""
Unit tests for model training functions
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

def test_data_split():
    """Test that train/test split works correctly"""
    # Create dummy data
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    
    # Test split sizes
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    assert len(X_train) == 80  # 80% of 100 = 80
    assert len(X_test) == 20   # 20% of 100 = 20
    assert len(y_train) == 80
    assert len(y_test) == 20
    
    print("✓ test_data_split passed")

def test_model_training():
    """Test that a model can be trained"""
    from sklearn.linear_model import LogisticRegression
    
    # Create dummy data
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    
    # Make predictions
    predictions = model.predict(X)
    
    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})  # Should be 0 or 1
    
    # Should have probability estimates
    proba = model.predict_proba(X)
    assert proba.shape == (50, 2)  # 50 samples, 2 classes
    assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1
    
    print("✓ test_model_training passed")

def test_metrics_calculation():
    """Test evaluation metrics calculation"""
    # Create dummy predictions
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1])
    
    # Calculate metrics manually
    accuracy = np.mean(y_true == y_pred)
    precision = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_pred == 1)
    recall = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1)
    
    assert accuracy == 0.75
    assert precision == 0.75
    assert recall == 0.75
    
    print("✓ test_metrics_calculation passed")

if __name__ == "__main__":
    test_data_split()
    test_model_training()
    test_metrics_calculation()
    print("\n✅ All tests passed!")