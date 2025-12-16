# setup.sh
#!/bin/bash

# Setup script for Credit Risk Model API

echo "Setting up Credit Risk Model API..."

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p models
mkdir -p data/processed
mkdir -p tests

# Check if model exists, if not create dummy model
if [ ! -f "models/best_model.pkl" ]; then
    echo "Creating dummy model for testing..."
    python -c "
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# Create dummy model
model = LogisticRegression(random_state=42, max_iter=1000)
X_dummy = np.random.randn(100, 6)
y_dummy = np.random.randint(0, 2, 100)
model.fit(X_dummy, y_dummy)

# Create dummy scaler
scaler = StandardScaler()
scaler.fit(X_dummy)

# Save to files
joblib.dump(model, 'models/best_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print('✓ Dummy model and scaler created')
"
fi

# Run tests
echo "Running tests..."
python -m pytest tests/ -v

echo "Setup complete!"
echo ""
echo "To run the API locally:"
echo "  uvicorn src.api.main:app --reload"
echo ""
echo "To run with Docker:"
echo "  docker-compose up"