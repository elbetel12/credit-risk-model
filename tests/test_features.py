import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.target_engineering import calculate_rfm

def test_rfm_calculation_empty_df():
    """Test RFM calculation with empty dataframe"""
    df = pd.DataFrame(columns=['CustomerId', 'Amount', 'TransactionStartTime'])
    rfm = calculate_rfm(df)
    # The function returns rfm as a dataframe, which should be empty if input is empty
    # Wait, looking at src/target_engineering.py:
    # snapshot_date = df['TransactionStartTime'].max() will fail on empty df if not handled
    # rfm = df.groupby('CustomerId').agg(...)
    
    # Let's see how the implementation handles it. 
    # If it fails, that's a bug we should identify, but user asked to TEST it.
    # Usually empty input should return empty output or handle gracefully.
    try:
        rfm = calculate_rfm(df)
        assert len(rfm) == 0
    except Exception as e:
        # If it raises an error, we might want to suggest a fix or document it.
        # But for the purpose of the test, let's assume it should handle it.
        pytest.fail(f"calculate_rfm failed on empty dataframe: {e}")

def test_rfm_calculation_missing_values():
    """Test RFM calculation with missing values"""
    data = {
        'CustomerId': ['C1', 'C1', 'C2'],
        'Amount': [100, None, 200],
        'TransactionStartTime': ['2023-01-01', '2023-01-02', '2023-01-03']
    }
    df = pd.DataFrame(data)
    rfm = calculate_rfm(df)
    
    # Check that C1 still has values despite one missing Amount
    # Missing Amount in groupby agg sum() usually treats it as 0 or ignores it.
    assert 'C1' in rfm['CustomerId'].values
    assert rfm.loc[rfm['CustomerId'] == 'C1', 'Monetary'].iloc[0] == 100.0
    assert rfm.loc[rfm['CustomerId'] == 'C2', 'Monetary'].iloc[0] == 200.0

def test_rfm_calculation_extreme_values():
    """Test feature engineering with extreme values"""
    data = {
        'CustomerId': ['C1', 'C2'],
        'Amount': [1e12, -1e12],
        'TransactionStartTime': ['2023-01-01', '2023-01-01']
    }
    df = pd.DataFrame(data)
    rfm = calculate_rfm(df)
    
    assert rfm.loc[rfm['CustomerId'] == 'C1', 'Monetary'].iloc[0] == 1e12
    assert rfm.loc[rfm['CustomerId'] == 'C2', 'Monetary'].iloc[0] == -1e12

def test_feature_distributions():
    """Test that feature distributions are as expected"""
    # Create 100 transactions for 10 customers
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = {
        'CustomerId': [f'C{i%10}' for i in range(100)],
        'Amount': np.random.uniform(10, 1000, 100),
        'TransactionStartTime': dates
    }
    df = pd.DataFrame(data)
    rfm = calculate_rfm(df)
    
    # Recency should be >= 0
    assert (rfm['Recency'] >= 0).all()
    # Frequency should be 10 for each (100 total / 10 customers)
    assert (rfm['Frequency'] == 10).all()
    # Monetary should be > 0
    assert (rfm['Monetary'] > 0).all()
    
    # Check if snapshot date logic works (min recency should be 0 since last date is max date)
    assert rfm['Recency'].min() == 0
