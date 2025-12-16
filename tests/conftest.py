import pytest
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

@pytest.fixture
def sample_input_data():
    """Sample input data for testing"""
    return {
        "features": [1.0, 2.0, 3.0, 4.0, 5.0],
        "customer_id": "test_123"
    }

@pytest.fixture
def sample_batch_input():
    """Sample batch input for testing"""
    return {
        "data": [
            {"feature1": 1.0, "feature2": 2.0},
            {"feature1": 3.0, "feature2": 4.0}
        ]
    }