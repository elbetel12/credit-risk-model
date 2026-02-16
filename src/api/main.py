# src/api/main.py
"""
FastAPI Application for Credit Risk Prediction
"""

import logging
import sys
import os
from typing import List, Optional

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# Import Pydantic models
from .pydantic_models import (
    PredictionRequest, PredictionResponse, HealthResponse,
    BatchPredictionRequest, BatchPredictionResponse
)
from .simplified_models import (
    SimplifiedPredictionRequest, SimplifiedPredictionResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting credit risk of customers",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and scaler
model = None
scaler = None
model_type = None
feature_names = [
    'total_amount', 'avg_amount', 'log_total_amount',
    'transaction_count', 'log_transaction_count', 'avg_txn_size',
    'days_since_last_transaction', 'days_since_first_transaction',
    'transaction_frequency', 'avg_days_between_transactions',
    'active_recently', 'customer_tenure_weeks',
    'monetary_per_day', 'amount_category_code', 'tx_count_category_code'
]



def load_model():
    """Load the trained model and scaler"""
    global model, scaler, model_type, feature_names
    
    try:
        # Load model
        model_path = "models/best_model.pkl"
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
            
        model = joblib.load(model_path)
        
        # Load feature info to know what features model expects
        feature_info_path = "models/feature_info.pkl"
        if os.path.exists(feature_info_path):
            feature_info = joblib.load(feature_info_path)
            feature_names = feature_info.get('feature_names', [])
            logger.info(f"Loaded {len(feature_names)} expected features from feature_info.pkl")
        else:
            logger.warning("feature_info.pkl not found, using default features")
            # Fallback to what we think features are
            feature_names = [
                'total_amount', 'avg_amount', 'log_total_amount',
                'transaction_count', 'log_transaction_count', 'avg_txn_size',
                'days_since_last_transaction', 'days_since_first_transaction',
                'transaction_frequency', 'avg_days_between_transactions',
                'active_recently', 'customer_tenure_weeks',
                'monetary_per_day', 'amount_category_code', 'tx_count_category_code'
            ]
        
        # Load scaler
        scaler_path = "models/scaler.pkl"
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            logger.warning("Scaler not found, using model without scaling")
        
        model_type = type(model).__name__
        logger.info(f"✓ Model loaded: {model_type}")
        logger.info(f"✓ Expected features: {len(feature_names)}")
        logger.info(f"✓ Scaler loaded: {scaler is not None}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False

def preprocess_features(data: PredictionRequest) -> np.ndarray:
    """Preprocess features for prediction - matches training feature engineering"""
    # Create feature dictionary - match EXACTLY what model expects
    features = {}
    
    # REQUIRED FIELDS (must be in request)
    features['total_amount'] = data.total_amount
    features['transaction_count'] = data.transaction_count
    features['days_since_last_transaction'] = data.days_since_last_transaction
    features['days_since_first_transaction'] = data.days_since_first_transaction
    
    # OPTIONAL FIELDS - use if provided, otherwise calculate
    # Monetary features
    features['avg_amount'] = data.avg_amount if data.avg_amount is not None else data.total_amount / max(1, data.transaction_count)
    features['log_total_amount'] = data.log_total_amount if data.log_total_amount is not None else np.log1p(max(0, data.total_amount))
    
    # Transaction features
    features['log_transaction_count'] = data.log_transaction_count if data.log_transaction_count is not None else np.log1p(max(0, data.transaction_count))
    features['avg_txn_size'] = data.avg_txn_size if data.avg_txn_size is not None else data.total_amount / max(1, data.transaction_count)
    
    # Date-based features
    transaction_period = data.days_since_first_transaction - data.days_since_last_transaction
    transaction_period = max(transaction_period, 1)  # Avoid division by zero
    
    features['transaction_frequency'] = data.transaction_frequency if data.transaction_frequency is not None else data.transaction_count / transaction_period
    features['avg_days_between_transactions'] = data.avg_days_between_transactions if data.avg_days_between_transactions is not None else transaction_period / max(1, data.transaction_count)
    features['active_recently'] = data.active_recently if data.active_recently is not None else (1 if data.days_since_last_transaction <= 30 else 0)
    features['customer_tenure_weeks'] = data.customer_tenure_weeks if data.customer_tenure_weeks is not None else data.days_since_first_transaction / 7
    
    # Derived features
    features['monetary_per_day'] = data.monetary_per_day if data.monetary_per_day is not None else data.total_amount / transaction_period
    
    # Amount categories (simplified calculation)
    if data.amount_category_code is not None:
        features['amount_category_code'] = data.amount_category_code
    else:
        if data.total_amount <= 100:
            features['amount_category_code'] = 0  # very_low
        elif data.total_amount <= 500:
            features['amount_category_code'] = 1  # low
        elif data.total_amount <= 2000:
            features['amount_category_code'] = 2  # medium
        else:
            features['amount_category_code'] = 3  # high
    
    # Transaction count categories
    if data.tx_count_category_code is not None:
        features['tx_count_category_code'] = data.tx_count_category_code
    else:
        if data.transaction_count <= 5:
            features['tx_count_category_code'] = 0  # few
        elif data.transaction_count <= 20:
            features['tx_count_category_code'] = 1  # moderate
        elif data.transaction_count <= 50:
            features['tx_count_category_code'] = 2  # many
        else:
            features['tx_count_category_code'] = 3  # very_many
    
    # Convert to DataFrame with CORRECT column order (as model expects)
    # Use the actual feature_names loaded from feature_info.pkl
    features_df = pd.DataFrame([features])
    
    # Ensure all expected features are present
    missing_features = [f for f in feature_names if f not in features_df.columns]
    if missing_features:
        logger.warning(f"Missing features in request: {missing_features}")
        # Add missing features with default values
        for feat in missing_features:
            features_df[feat] = 0
    
    # Reorder columns to match model expectations
    features_df = features_df[feature_names]
    
    # Apply scaling if scaler exists
    if scaler is not None:
        features_scaled = scaler.transform(features_df)
    else:
        features_scaled = features_df.values
    
    logger.info(f"Processed features. Shape: {features_scaled.shape}")
    return features_scaled

def get_risk_category(probability: float) -> str:
    """Convert probability to risk category"""
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("🚀 Starting Credit Risk Prediction API...")
    logger.info(f"Current directory: {os.getcwd()}")
    
    if os.path.exists('models'):
        logger.info(f"Files in models/: {os.listdir('models')}")
    else:
        logger.warning("Models directory NOT found!")
    
    success = load_model()
    if not success:
        logger.error("Failed to load model on startup")
    else:
        logger.info("✅ Model loaded successfully on startup")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Credit Risk Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    model_loaded = model is not None
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        model_type=model_type if model_loaded else None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest):
    """Predict credit risk for a single customer"""
    try:
        # Check if model is loaded
        if model is None:
            logger.error("Model not loaded!")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        logger.info(f"Received prediction request for: {request.CustomerId}")
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Model has predict_proba: {hasattr(model, 'predict_proba')}")
        
        # Preprocess features
        logger.info("Preprocessing features...")
        features = preprocess_features(request)
        logger.info(f"Features shape: {features.shape}")
        logger.info(f"Features dtype: {features.dtype}")
        
        # Make prediction
        try:
            logger.info("Making prediction...")
            if hasattr(model, 'predict_proba'):
                logger.info("Using predict_proba method")
                probs = model.predict_proba(features)
                # Use robust indexing that works for both numpy arrays and lists
                probability = probs[0][1] if isinstance(probs, (list, np.ndarray)) and len(probs) > 0 else 0.5
            elif hasattr(model, 'predict'):
                logger.info("Using predict method")
                prediction = model.predict(features)
                probability = float(prediction[0]) if isinstance(prediction, (list, np.ndarray)) else float(prediction)
            else:
                logger.error("Model has no predict or predict_proba method!")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Model cannot make predictions"
                )
            
            logger.info(f"Prediction probability: {probability}")
            
            # Determine if high risk (threshold at 0.5)
            is_high_risk = probability >= 0.5
            
            # Create response
            response = PredictionResponse(
                customer_id=request.CustomerId,
                is_high_risk=bool(is_high_risk),
                risk_probability=float(probability),
                risk_category=get_risk_category(float(probability))
            )
            
            logger.info(f"Prediction successful for {request.CustomerId}")
            return response
            
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /predict: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
    

@app.post("/predict/simple", response_model=SimplifiedPredictionResponse, tags=["Predictions"])
async def predict_simple(request: SimplifiedPredictionRequest):
    """
    Simplified prediction endpoint for dashboard with user-friendly features.
    Accepts: recency, frequency, monetary_volatility, avg_amount, weekend_ratio
    Returns: risk_score, category, credit_limit, etc.
    """
    try:
        # Check if model is loaded
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        logger.info(f"Received simplified prediction request for: {request.customer_id}")
        
        # Convert simplified features to model-expected format
        # We'll create realistic transaction-based features from the simplified inputs
        
        # Estimate transaction count from monthly frequency
        # Assume customer has been active for ~6 months on average
        estimated_months = 6
        transaction_count = int(request.frequency * estimated_months)
        transaction_count = max(transaction_count, 1)  # At least 1 transaction
        
        # Calculate total amount from average and count
        total_amount = request.avg_amount * transaction_count
        
        # Use recency directly as days_since_last_transaction
        days_since_last_transaction = request.recency
        
        # Estimate days_since_first_transaction (tenure)
        # If very recent, tenure is short; if not recent, tenure is longer
        days_since_first_transaction = max(180, days_since_last_transaction + (estimated_months * 30))
        
        # Create PredictionRequest with estimated features
        traditional_request = PredictionRequest(
            CustomerId=request.customer_id,
            total_amount=total_amount,
            transaction_count=transaction_count,
            days_since_last_transaction=days_since_last_transaction,
            days_since_first_transaction=days_since_first_transaction,
            avg_amount=request.avg_amount
        )
        
        # Preprocess features using existing function
        features = preprocess_features(traditional_request)
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(features)
            probability = probs[0][1] if isinstance(probs, (list, np.ndarray)) and len(probs) > 0 else 0.5
        elif hasattr(model, 'predict'):
            prediction = model.predict(features)
            probability = float(prediction[0]) if isinstance(prediction, (list, np.ndarray)) else float(prediction)
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model cannot make predictions"
            )
        
        # Determine risk category
        category = get_risk_category(float(probability))
        
        # Calculate credit limit based on risk score
        # Lower risk = higher credit limit
        base_limit = request.avg_amount * 10  # Base: 10x average transaction
        
        if probability < 0.3:  # Low risk
            credit_limit = base_limit * 1.5
        elif probability < 0.7:  # Medium risk
            credit_limit = base_limit * 1.0
        else:  # High risk
            credit_limit = base_limit * 0.5
        
        # Cap credit limit between reasonable bounds
        credit_limit = max(50000, min(credit_limit, 5000000))  # 50K to 5M UGX
        
        # Calculate interest rate (higher risk = higher rate)
        # Range: 8% (low risk) to 25% (high risk)
        base_rate = 8.0
        risk_premium = probability * 17.0  # 0-17% risk premium
        interest_rate = base_rate + risk_premium
        
        # Calculate expected loss
        # Expected Loss = PD × Exposure × LGD
        # Assuming Loss Given Default (LGD) = 50%
        lgd = 0.5
        expected_loss = probability * credit_limit * lgd
        
        # Create response
        response = SimplifiedPredictionResponse(
            customer_id=request.customer_id,
            risk_score=float(probability),
            category=category,
            credit_limit=float(credit_limit),
            pd=float(probability),
            interest_rate=float(interest_rate),
            expected_loss=float(expected_loss)
        )
        
        logger.info(f"Simplified prediction successful: {category} risk ({probability:.2%})")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /predict/simple: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """Predict credit risk for multiple customers"""
    try:
        # Check if model is loaded
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        predictions = []
        high_risk_count = 0
        
        for customer in request.customers:
            try:
                # Preprocess features
                features = preprocess_features(customer)
                
                # Make prediction
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(features)
                    probability = probs[0][1] if isinstance(probs, (list, np.ndarray)) and len(probs) > 0 else 0.5
                else:
                    prediction = model.predict(features)
                    probability = float(prediction[0]) if isinstance(prediction, (list, np.ndarray)) else float(prediction)
                
                is_high_risk = probability >= 0.5
                if is_high_risk:
                    high_risk_count += 1
                
                prediction = PredictionResponse(
                    customer_id=customer.CustomerId,
                    is_high_risk=bool(is_high_risk),
                    risk_probability=float(probability),
                    risk_category=get_risk_category(float(probability))
                )
                
                predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"Error processing customer {customer.CustomerId}: {e}")
                # Continue with other customers even if one fails
                continue
        
        response = BatchPredictionResponse(
            predictions=predictions,
            total_customers=len(request.customers),
            high_risk_count=high_risk_count
        )
        
        logger.info(f"Batch prediction completed: "
                   f"processed {len(predictions)} customers, "
                   f"{high_risk_count} high risk")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /predict/batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.get("/features", tags=["Info"])
async def get_features():
    """Get expected features for prediction"""
    return {
        "expected_features": feature_names,
        "description": "Features expected by the model for prediction"
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    # Load model before starting server
    load_model()
    
    # Start the server
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )