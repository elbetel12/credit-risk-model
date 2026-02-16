# src/api/simplified_models.py
"""
Simplified Pydantic models for user-friendly dashboard interface
These models use intuitive feature names that make sense to business users
"""

from pydantic import BaseModel, Field
from typing import Optional


class SimplifiedPredictionRequest(BaseModel):
    """Simplified request model for dashboard - user-friendly feature names"""
    
    # Optional customer ID
    customer_id: Optional[str] = Field(default="DASHBOARD_USER", description="Customer ID")
    
    # User-friendly features (RFM-based)
    recency: float = Field(..., description="Days since last transaction (0-365)", ge=0, le=365)
    frequency: float = Field(..., description="Monthly transaction frequency (0-100)", ge=0, le=100)
    monetary_volatility: float = Field(..., description="Transaction amount volatility (0-1)", ge=0, le=1)
    avg_amount: float = Field(..., description="Average transaction amount in UGX", ge=0)
    weekend_ratio: float = Field(..., description="Proportion of weekend transactions (0-1)", ge=0, le=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "DASH001",
                "recency": 15.0,
                "frequency": 25.0,
                "monetary_volatility": 0.35,
                "avg_amount": 75000.0,
                "weekend_ratio": 0.25
            }
        }


class SimplifiedPredictionResponse(BaseModel):
    """Simplified response model for dashboard"""
    
    customer_id: str = Field(..., description="Customer ID")
    risk_score: float = Field(..., description="Risk score (probability of default, 0-1)")
    category: str = Field(..., description="Risk category (Low/Medium/High)")
    credit_limit: float = Field(..., description="Recommended credit limit in UGX")
    pd: float = Field(..., description="Probability of default (same as risk_score)")
    interest_rate: float = Field(..., description="Recommended interest rate (%)")
    expected_loss: float = Field(..., description="Expected loss in UGX")
    
    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "DASH001",
                "risk_score": 0.35,
                "category": "Medium",
                "credit_limit": 250000.0,
                "pd": 0.35,
                "interest_rate": 12.5,
                "expected_loss": 43750.0
            }
        }
