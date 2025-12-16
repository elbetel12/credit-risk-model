import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock the model loading for tests
@pytest.fixture
def mock_model():
    class MockModel:
        def predict(self, X):
            return [0] * len(X)
        
        def predict_proba(self, X):
            return [[0.7, 0.3]] * len(X)
    
    return MockModel()

def test_health_endpoint():
    """Test the health endpoint"""
    # Mock the app import to avoid model loading issues
    import app.main
    app.main.model_loaded = True  # Mock model as loaded
    
    from app.main import app
    client = TestClient(app)
    
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["healthy", "unhealthy"]

def test_predict_endpoint():
    """Test the predict endpoint"""
    # This would be a more complete test with mocked dependencies
    pass