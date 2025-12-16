# src/api/pydantic_models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint - Updated for new features"""
    CustomerId: str = Field(..., description="Customer ID")
    
    # Monetary features
    total_amount: float = Field(..., description="Total transaction amount")
    avg_amount: Optional[float] = Field(None, description="Average transaction amount")
    log_total_amount: Optional[float] = Field(None, description="Log of total amount")
    
    # Transaction features
    transaction_count: int = Field(..., description="Number of transactions")
    log_transaction_count: Optional[float] = Field(None, description="Log of transaction count")
    avg_txn_size: Optional[float] = Field(None, description="Average transaction size")
    
    # Date-based features (CRITICAL - from updated training)
    days_since_last_transaction: float = Field(..., description="Days since last transaction")
    days_since_first_transaction: float = Field(..., description="Days since first transaction")
    transaction_frequency: Optional[float] = Field(None, description="Transactions per day")
    avg_days_between_transactions: Optional[float] = Field(None, description="Average days between transactions")
    active_recently: Optional[int] = Field(None, description="1 if active in last 30 days, else 0")
    customer_tenure_weeks: Optional[float] = Field(None, description="Customer tenure in weeks")
    
    # Derived features
    monetary_per_day: Optional[float] = Field(None, description="Monetary value per day")
    amount_category_code: Optional[int] = Field(None, description="Amount category code (0-3)")
    tx_count_category_code: Optional[int] = Field(None, description="Transaction count category code (0-3)")
    
    class Config:
        # Make it flexible for optional fields
        extra = "ignore"
        json_schema_extra = {
            "example": {
                "CustomerId": "CUST001",
                "total_amount": 1500.75,
                "transaction_count": 12,
                "days_since_last_transaction": 7.0,
                "days_since_first_transaction": 365.0,
                # Optional fields can be omitted
            }
        }

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    customer_id: str = Field(..., description="Customer ID")
    is_high_risk: bool = Field(..., description="Whether customer is high risk")
    risk_probability: float = Field(..., description="Probability of being high risk (0-1)")
    risk_category: str = Field(..., description="Risk category (Low/Medium/High)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "CUST001",
                "is_high_risk": False,
                "risk_probability": 0.25,
                "risk_category": "Low"
            }
        }

# Keep other models (HealthResponse, BatchPredictionRequest, etc.) the same

class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_type: Optional[str] = Field(None, description="Type of loaded model")
    version: str = Field(..., description="API version")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_type": "RandomForestClassifier",
                "version": "1.0.0"
            }
        }

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    customers: List[PredictionRequest] = Field(..., description="List of customer data")
    
    class Config:
        json_schema_extra = {
            "example": {
                "customers": [
                    {
                        "CustomerId": "CUST001",
                        "total_amount": 1500.75,
                        "transaction_count": 12,
                        "avg_transaction": 125.06,
                        "days_since_last_transaction": 7.0,
                        "days_since_first_transaction": 365.0,
                        "transaction_frequency": 0.033
                    }
                ]
            }
        }

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_customers: int = Field(..., description="Total number of customers processed")
    high_risk_count: int = Field(..., description="Number of high-risk customers")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "customer_id": "CUST001",
                        "is_high_risk": False,
                        "risk_probability": 0.25,
                        "risk_category": "Low"
                    }
                ],
                "total_customers": 1,
                "high_risk_count": 0
            }
        }