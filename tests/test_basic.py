import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def test_basic():
    """Basic test to ensure environment works"""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)
    
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(y_test), 2)
    
    print("All basic tests passed")
    return True

if __name__ == "__main__":
    test_basic()
