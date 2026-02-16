# Credit Risk Dashboard - MVP

## 📊 Overview
This is a **Minimum Viable Product (MVP)** Streamlit dashboard for the BNPL Credit Risk Assessment system. It provides an interactive interface for finance stakeholders to assess customer credit risk in real-time.

## ✨ Features

### Core Functionality
- ✅ **Interactive Input Form**: All 5 model features with appropriate widgets
- ✅ **Real-time Risk Assessment**: Instant predictions via FastAPI backend
- ✅ **Color-Coded Results**: Visual risk categorization (Low/Medium/High)
- ✅ **Credit Limit Recommendations**: Automated credit limit calculations
- ✅ **Sample Data Loading**: Quick demo with pre-filled customer data
- ✅ **API Details View**: Expandable request/response JSON viewer
- ✅ **Error Handling**: Graceful handling of API connection issues
- ✅ **Loading States**: User feedback during API calls

### Professional Design
- 🎨 Clean, modern interface optimized for finance stakeholders
- 📱 Responsive 3-column layout for key metrics
- 🎯 Color-coded risk categories with emojis
- 💳 Currency formatting for UGX amounts
- 📊 Percentage formatting for risk scores

## 🚀 Quick Start

### Option 1: Automated Start (Recommended)
```bash
# Double-click or run from command prompt:
start_dashboard.bat
```
This will automatically:
- Start the FastAPI backend on port 8000
- Start the Streamlit dashboard on port 8501
- Open the dashboard in your browser

### Option 2: Manual Start

#### Step 1: Start the API Backend
```bash
# Terminal 1: Start the FastAPI server
cd c:\credit-risk-model
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Verify the API is running by visiting `http://localhost:8000/docs`

#### Step 2: Start the Dashboard
```bash
# Terminal 2: Start the Streamlit dashboard
cd c:\credit-risk-model
streamlit run dashboard/app.py
```

The dashboard will automatically open in your browser at `http://localhost:8501`

## 📸 Taking Screenshots

For your submission, capture these key screens:

1. **Main Dashboard** - Show the welcome screen with feature descriptions
2. **Input Form** - Sidebar with all 5 input features
3. **Results Display** - All three result cards (Risk Score, Category, Credit Limit)
4. **Sample Customer** - Click "Load Sample Customer" and show results
5. **API Details** - Expand the "View API Request/Response Details" section

## 🎯 Model Features

The dashboard accepts 5 key features:

| Feature | Description | Input Type | Range |
|---------|-------------|------------|-------|
| **Recency** | Days since last transaction | Number Input | 0-365 |
| **Frequency** | Monthly transaction count | Number Input | 0-100 |
| **Monetary Volatility** | Transaction amount variability | Slider | 0.0-1.0 |
| **Average Amount** | Average transaction (UGX) | Number Input | 1,000-5,000,000 |
| **Weekend Ratio** | Proportion of weekend transactions | Slider | 0.0-1.0 |

## 🎨 Risk Categories

| Category | Risk Score Range | Color | Credit Action |
|----------|------------------|-------|---------------|
| **Low** ✅ | 0% - 33% | Green (#00CC96) | High credit limit |
| **Medium** ⚠️ | 33% - 67% | Orange (#FFA15A) | Moderate credit limit |
| **High** 🚨 | 67% - 100% | Red (#EF553B) | Low/No credit limit |

## 📊 Sample Customer Data

The dashboard includes a "Load Sample Customer" button with these default values:
- Recency: 15 days
- Frequency: 25 transactions/month
- Monetary Volatility: 0.35
- Average Amount: UGX 75,000
- Weekend Ratio: 0.25

Expected Result: **Medium Risk** (~35% risk score)

## 🔧 API Integration

The dashboard connects to the FastAPI backend at:
```
POST http://localhost:8000/predict/simple
```

**Request Format:**
```json
{
  "customer_id": "DASH001",
  "recency": 15.0,
  "frequency": 25.0,
  "monetary_volatility": 0.35,
  "avg_amount": 75000.0,
  "weekend_ratio": 0.25
}
```

**Response Format:**
```json
{
  "risk_score": 0.35,
  "category": "Medium",
  "credit_limit": 250000,
  "pd": 0.35,
  "interest_rate": 12.5,
  "expected_loss": 43750
}
```

## ⚠️ Troubleshooting

### "Cannot connect to API" Error
- ✅ Ensure FastAPI is running on `http://localhost:8000`
- ✅ Check if port 8000 is available
- ✅ Verify the API endpoint: `http://localhost:8000/docs`

### Dashboard won't start
- ✅ Install Streamlit: `pip install streamlit`
- ✅ Check Python version (3.8+)
- ✅ Try: `python -m streamlit run dashboard/app.py`

### Port already in use
- ✅ Streamlit uses port 8501 by default
- ✅ Change port: `streamlit run dashboard/app.py --server.port 8502`

## 🚧 Future Enhancements

This is an MVP version. Planned features for full release:
- 📈 Historical trend analysis
- 📊 Batch customer processing
- 💾 Results export (CSV/PDF)
- 📉 Interactive risk distribution charts
- 🔍 Customer segmentation analysis
- 📱 Mobile-responsive design improvements
- 🔐 User authentication
- 📝 Audit logging

## 📝 Notes for Submission

**Model Performance:**
- Algorithm: Random Forest Classifier
- Accuracy: 89%
- Features: 5 (behavioral + financial)
- Training: Historical transaction data

**Technology Stack:**
- Frontend: Streamlit 1.29.0
- Backend: FastAPI 0.104.1
- Model: Scikit-learn 1.3.2
- Tracking: MLflow 2.10.1

**Development Status:**
- ✅ MVP Complete
- ✅ API Integration Working
- ✅ Error Handling Robust
- 🚧 Advanced Analytics (Coming Soon)

---

**Created for:** Week 4 - Second Interim Submission  
**Date:** February 16, 2026  
**Project:** Credit Risk Assessment - BNPL Model
