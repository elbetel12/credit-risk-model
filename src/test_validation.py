# test_validation.py
import requests
import json

# Test different payloads
payloads = [
    {
        "CustomerId": "TEST001",
        "total_amount": 1500.75,
        "transaction_count": 12,
        "avg_amount": 125.06,
        "days_since_last_transaction": 7.0,
        "days_since_first_transaction": 365.0,
        "std_amount": 50.0
    },
    {
        "CustomerId": "TEST001",
        "total_amount": 1500.75,
        "transaction_count": 12,
        "avg_transaction": 125.06,
        "days_since_last_transaction": 7.0,
        "days_since_first_transaction": 365.0,
        "transaction_frequency": 0.033
    }
]

for i, payload in enumerate(payloads):
    print(f"\n{'='*50}")
    print(f"Testing payload {i+1}:")
    print(json.dumps(payload, indent=2))
    
    response = requests.post("http://localhost:8000/predict", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")