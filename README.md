## Credit Scoring Business Understanding

## 1. Basel II Accord's Influence on Model Requirements

The Basel II Accord emphasizes three pillars: minimum capital requirements, supervisory review, and market discipline. This directly impacts our model development because:
- **Interpretability is crucial**: Regulators require models to be explainable to ensure proper risk measurement
- **Documentation is mandatory**: All modeling assumptions, methodologies, and validations must be thoroughly documented
- **Risk-weighted assets calculation**: Our model's outputs may directly influence capital allocation decisions

## 2. Proxy Variable Necessity and Risks

**Why a proxy variable is necessary:**
- The eCommerce transaction data lacks direct loan performance labels
- We need to infer credit risk from behavioral patterns
- RFM (Recency, Frequency, Monetary) analysis provides a reasonable approximation of customer engagement, which correlates with repayment likelihood

**Potential business risks:**
- **False positives**: Labeling good customers as high-risk, losing potential revenue
- **False negatives**: Approving loans to truly high-risk customers, increasing defaults
- **Model drift**: Behavioral patterns may change over time, requiring regular retraining
- **Regulatory scrutiny**: Using proxies requires strong justification and ongoing validation

## 3. Model Choice Trade-offs

**Simple, Interpretable Models (Logistic Regression with WoE):**
- ✅ **Advantages**: Easily explainable to regulators, transparent coefficients, stable predictions
- ✅ **Basel II compliance**: Better aligns with regulatory requirements for explainability
- ❌ **Disadvantages**: May capture fewer complex patterns, potentially lower predictive power

**Complex Models (Gradient Boosting):**
- ✅ **Advantages**: Higher predictive accuracy, captures non-linear relationships
- ❌ **Disadvantages**: "Black box" nature makes regulatory approval challenging
- ❌ **Risk management**: Harder to justify decisions to stakeholders and regulators

**Recommended Approach**: Start with Logistic Regression + WoE for regulatory compliance, then explore Gradient Boosting for comparison while maintaining thorough documentation.

# Credit Risk Model - Bati Bank BNPL Service

![CI](https://github.com/elbetel12/credit-risk-model/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)
![Status](https://img.shields.io/badge/Status-Interim_2-orange.svg)

## 📋 Project Overview
An end-to-end production-ready credit scoring system for Bati Bank's Buy-Now-Pay-Later (BNPL) service. This project implements a full machine learning pipeline from data engineering to a containerized API deployment with CI/CD integration.

### Interim Submission 2 Progress (Tonight's Deadline)
| Task | Status | Completion |
| :--- | :---: | :---: |
| Feature Engineering (RFM + Time) | ✅ | 100% |
| Model Training (Random Forest) | ✅ | 100% |
| SHAP Explainability | ✅ | 100% |
| FastAPI Endpoint Implementation | ✅ | 100% |
| Unit Testing (10/10 tests) | ✅ | 100% |
| CI/CD Pipeline (GitHub Actions) | ✅ | 100% |
| Streamlit Dashboard MVP | 🔄 | 80% |

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Git
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/elbetel12/credit-risk-model.git
cd credit-risk-model

Set up virtual environment:

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

bash
pip install -r requirements.txt
Download data:
Place your transaction data in data/raw/transactions.csv

Running the Project
Exploratory Data Analysis:

bash
jupyter notebook notebooks/eda.ipynb
Process data and create target variable:

bash
python src/target_engineering.py
python src/data_processing.py
Train models:

bash
python src/train.py
Run the API locally:

bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
Docker Deployment
Build and run with Docker Compose:

bash
docker-compose up --build
Access the API:

API: http://localhost:8000

Documentation: http://localhost:8000/docs

MLflow UI: http://localhost:5000

Running Tests
bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_data_processing.py -v

# Run with coverage
pytest --cov=src tests/
📁 Project Structure
text
credit-risk-model/
├── data/              # Data directory (gitignored)
├── notebooks/         # Jupyter notebooks for EDA
├── src/              # Source code
├── tests/            # Unit tests
├── models/           # Trained models (gitignored)
├── Dockerfile        # Docker configuration
├── docker-compose.yml # Multi-container setup
└── requirements.txt  # Python dependencies
🔧 Configuration
Environment Variables
Create a .env file in the root directory:

env
ENVIRONMENT=development
LOG_LEVEL=DEBUG
MODEL_PATH=models/best_model.pkl
MLFLOW_TRACKING_URI=http://localhost:5000
📊 Model Details
Features Used
RFM (Recency, Frequency, Monetary) metrics

Transaction patterns (time, day, channel)

Customer behavior aggregates

Product category trends

Models Implemented
Logistic Regression (interpretable baseline)

Random Forest (balanced performance)

Gradient Boosting (high accuracy)

Evaluation Metrics
ROC-AUC Score

Precision & Recall

F1 Score

Confusion Matrix

🧪 Testing
Run the test suite:

bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests with coverage
pytest --cov=src --cov-report=html
📈 Monitoring & Logging
MLflow for experiment tracking

Built-in logging to app.log

Health check endpoints

Performance metrics tracking

🤝 Contributing
Fork the repository

Create a feature branch

Add tests for new functionality

Ensure all tests pass

Submit a pull request


# Task 6: Model Deployment and Continuous Integration

## Overview
This task packages the trained credit risk model into a containerized API and sets up a CI/CD pipeline for automated testing.

## Structure
src/api/
├── main.py # FastAPI application
├── pydantic_models.py # Request/response models
Dockerfile # Container configuration
docker-compose.yml # Multi-container setup
.github/workflows/ci.yml # CI/CD pipeline

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt

2. Ensure model files exist
Make sure you have:

models/best_model.pkl - Trained model

models/scaler.pkl - Feature scaler

3. Run the API locally
uvicorn src.api.main:app --reload
4. Access the API
API: http://localhost:8000

Docs: http://localhost:8000/docs

Health: http://localhost:8000/health

Docker Deployment
Build and run with Docker
# Build the image
docker build -t credit-risk-api .

# Run the container
docker run -p 8000:8000 credit-risk-api

Using Docker Compose
# Start all services
docker-compose up

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
API Endpoints
Health Check
GET /health 
Returns API status and model information.
Single Prediction
POST /predict
Predict credit risk for a single customer.
Batch Prediction
POST /predict/batch
Predict credit risk for multiple customers.
Available Features
GET /features
Get list of features expected by the model.

CI/CD Pipeline
The CI/CD pipeline runs on every push to main/master branch:

Linting: Checks code style with flake8 and black

Testing: Runs unit tests with pytest

Building: Builds Docker image

Deployment: Deploys to container registry (if on main branch)
Testing
Run tests locally
bash
pytest tests/ -v
Run with coverage
bash
pytest tests/ -v --cov=src --cov-report=html
Notes
The API loads the model from models/best_model.pkl on startup

Feature scaling is applied if models/scaler.pkl exists

Default prediction threshold is 0.5 for high risk classification

text

## Summary

You now have a complete Task 6 implementation with:

1. ✅ **FastAPI application** with proper endpoints
2. ✅ **Pydantic models** for request/response validation
3. ✅ **Docker containerization** with Dockerfile
4. ✅ **Multi-container setup** with docker-compose
5. ✅ **CI/CD pipeline** with GitHub Actions
6. ✅ **Unit tests** for API endpoints
7. ✅ **Setup scripts** and documentation

To run the complete setup:

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh

# Run the API
docker-compose up
The API will be available at http://localhost:8000 with interactive documentation at http://localhost:8000/docs


📝 License
This project is licensed under the MIT License.

