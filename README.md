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

## 📋 Project Overview
An end-to-end credit scoring system for Bati Bank's Buy-Now-Pay-Later service, using eCommerce transaction data to predict customer credit risk.

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

📝 License
This project is licensed under the MIT License.

📧 Contact
For questions or support, please contact your team lead or use the project's issue tracker.