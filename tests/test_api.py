import pytest
from fastapi.testclient import TestClient
import sys
import os
from unittest.mock import MagicMock, patch

# Add parent directory to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def client():
    """Create a test client for the API"""
    # Import inside fixture to avoid collection-time issues
    with patch('src.api.main.load_model', return_value=True):
        from src.api.main import app
        return TestClient(app)

@pytest.fixture
def mock_model():
    """Fixture to mock the model in main.py"""
    with patch('src.api.main.model') as mock:
        # Mock predict_proba to return 0.2 for class 1 (Low Risk)
        import numpy as np
        mock.predict_proba.return_value = np.array([[0.8, 0.2]])
        mock.predict.return_value = np.array([0])
        yield mock

def test_predict_missing_fields(client, mock_model):
    """Test /predict endpoint with missing fields"""
    # Missing 'total_amount' and 'transaction_count'
    payload = {
        "CustomerId": "CUST001",
        "days_since_last_transaction": 7.0,
        "days_since_first_transaction": 365.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity

def test_predict_negative_values(client, mock_model):
    """Test with negative values for recency/frequency"""
    # Note: Pydantic might not block negative unless specified, 
    # but the API logic or model might handle it.
    # Here we just test the API responds.
    payload = {
        "CustomerId": "CUST001",
        "total_amount": 1500.0,
        "transaction_count": -5,  # Negative
        "days_since_last_transaction": -7.0, # Negative
        "days_since_first_transaction": 365.0
    }
    # If there are no validators blocking negative values, it might return 200.
    # But usually we expect some validation or at least it doesn't crash.
    response = client.post("/predict", json=payload)
    assert response.status_code in [200, 422, 400]

def test_predict_extremely_large_values(client, mock_model):
    """Test with extremely large values"""
    payload = {
        "CustomerId": "CUST001",
        "total_amount": 1e15,
        "transaction_count": 1000000,
        "days_since_last_transaction": 7.0,
        "days_since_first_transaction": 365.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

def test_predict_wrong_data_types(client, mock_model):
    """Test with wrong data types (strings instead of numbers)"""
    payload = {
        "CustomerId": "CUST001",
        "total_amount": "not_a_number", # String
        "transaction_count": 12,
        "days_since_last_transaction": 7.0,
        "days_since_first_transaction": 365.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_api_status_codes(client, mock_model):
    """Test that API returns appropriate status codes"""
    # Health check
    response = client.get("/health")
    assert response.status_code == 200
    
    # Root
    response = client.get("/")
    assert response.status_code == 200
    
    # Invalid endpoint
    response = client.get("/invalid_endpoint")
    assert response.status_code == 404