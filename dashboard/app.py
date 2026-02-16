"""
Credit Risk Assessment Dashboard - MVP
BNPL Credit Risk Model - Interactive Dashboard

This Streamlit dashboard provides an interactive interface for assessing credit risk
using a Random Forest model with 89% accuracy. It connects to a FastAPI backend
to predict customer risk scores and credit limits.
"""

import streamlit as st
import requests
import json
from typing import Dict, Any, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

API_URL = "http://localhost:8000/predict/simple"

# Risk category thresholds and colors
RISK_CATEGORIES = {
    "Low": {"threshold": 0.33, "color": "#00CC96", "emoji": "✅"},
    "Medium": {"threshold": 0.67, "color": "#FFA15A", "emoji": "⚠️"},
    "High": {"threshold": 1.0, "color": "#EF553B", "emoji": "🚨"}
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_risk_category(risk_score: float) -> str:
    """Determine risk category based on risk score."""
    if risk_score < RISK_CATEGORIES["Low"]["threshold"]:
        return "Low"
    elif risk_score < RISK_CATEGORIES["Medium"]["threshold"]:
        return "Medium"
    else:
        return "High"


def get_category_color(category: str) -> str:
    """Get color code for risk category."""
    return RISK_CATEGORIES[category]["color"]


def get_category_emoji(category: str) -> str:
    """Get emoji for risk category."""
    return RISK_CATEGORIES[category]["emoji"]


def call_prediction_api(features: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """
    Call the FastAPI prediction endpoint.
    
    Args:
        features: Dictionary containing the 5 required features
        
    Returns:
        API response as dictionary or None if error
    """
    try:
        response = requests.post(
            API_URL,
            json=features,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Please ensure the FastAPI server is running on http://localhost:8000")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ API request timed out. Please try again.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API returned error: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return None


def get_sample_customer() -> Dict[str, float]:
    """Return sample customer data for demonstration."""
    return {
        "recency": 15.0,
        "frequency": 25.0,
        "monetary_volatility": 0.35,
        "avg_amount": 75000.0,
        "weekend_ratio": 0.25
    }


def format_currency(amount: float) -> str:
    """Format number as UGX currency."""
    return f"UGX {amount:,.0f}"


def format_percentage(value: float) -> str:
    """Format decimal as percentage."""
    return f"{value * 100:.1f}%"


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Credit Risk Dashboard",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .footer {
            text-align: center;
            color: #888;
            font-size: 0.9rem;
            margin-top: 3rem;
            padding: 1rem;
            border-top: 1px solid #ddd;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">💳 Credit Risk Assessment Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">BNPL Credit Risk Model - Powered by Random Forest (89% Accuracy)</div>', unsafe_allow_html=True)
    
    # Initialize session state for inputs
    if "inputs" not in st.session_state:
        st.session_state.inputs = {
            "recency": 30.0,
            "frequency": 15.0,
            "monetary_volatility": 0.3,
            "avg_amount": 50000.0,
            "weekend_ratio": 0.2
        }
    
    # ========================================================================
    # SIDEBAR - INPUT FORM
    # ========================================================================
    
    with st.sidebar:
        st.header("📊 Customer Input Features")
        st.markdown("---")
        
        # Sample customer button
        if st.button("🎯 Load Sample Customer", use_container_width=True):
            st.session_state.inputs = get_sample_customer()
            st.rerun()
        
        st.markdown("---")
        st.subheader("Transaction Behavior")
        
        # Recency
        recency = st.number_input(
            "📅 Days Since Last Transaction (Recency)",
            min_value=0,
            max_value=365,
            value=int(st.session_state.inputs["recency"]),
            help="Number of days since the customer's last transaction"
        )
        st.session_state.inputs["recency"] = float(recency)
        
        # Frequency
        frequency = st.number_input(
            "🔄 Monthly Transaction Frequency",
            min_value=0,
            max_value=100,
            value=int(st.session_state.inputs["frequency"]),
            help="Average number of transactions per month"
        )
        st.session_state.inputs["frequency"] = float(frequency)
        
        st.markdown("---")
        st.subheader("Financial Patterns")
        
        # Monetary Volatility
        monetary_volatility = st.slider(
            "📊 Amount Volatility",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.inputs["monetary_volatility"],
            step=0.01,
            help="Standard deviation of transaction amounts (0=consistent, 1=highly variable)"
        )
        st.session_state.inputs["monetary_volatility"] = monetary_volatility
        
        # Average Amount
        avg_amount = st.number_input(
            "💰 Average Transaction Amount (UGX)",
            min_value=1000,
            max_value=5000000,
            value=int(st.session_state.inputs["avg_amount"]),
            step=1000,
            help="Average amount per transaction in Ugandan Shillings"
        )
        st.session_state.inputs["avg_amount"] = float(avg_amount)
        
        # Weekend Ratio
        weekend_ratio = st.slider(
            "📅 Weekend Transaction Ratio",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.inputs["weekend_ratio"],
            step=0.01,
            help="Proportion of transactions made on weekends"
        )
        st.session_state.inputs["weekend_ratio"] = weekend_ratio
        
        st.markdown("---")
        
        # Submit button
        submit_button = st.button("🚀 Assess Credit Risk", type="primary", use_container_width=True)
    
    # ========================================================================
    # MAIN CONTENT - RESULTS DISPLAY
    # ========================================================================
    
    if submit_button:
        with st.spinner("🔄 Analyzing customer data..."):
            # Prepare features for API
            features = {
                "recency": st.session_state.inputs["recency"],
                "frequency": st.session_state.inputs["frequency"],
                "monetary_volatility": st.session_state.inputs["monetary_volatility"],
                "avg_amount": st.session_state.inputs["avg_amount"],
                "weekend_ratio": st.session_state.inputs["weekend_ratio"]
            }
            
            # Call API
            result = call_prediction_api(features)
            
            if result:
                # Store result in session state
                st.session_state.last_result = result
                st.session_state.last_features = features
                
                st.success("✅ Risk assessment completed successfully!")
                
                # Display results in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 📊 Risk Score")
                    risk_score = result.get("risk_score", result.get("pd", 0))
                    category = get_risk_category(risk_score)
                    color = get_category_color(category)
                    
                    st.markdown(
                        f'<div style="background-color: {color}; padding: 2rem; border-radius: 10px; text-align: center;">'
                        f'<h1 style="color: white; margin: 0;">{format_percentage(risk_score)}</h1>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    st.metric("Probability of Default", format_percentage(risk_score))
                
                with col2:
                    st.markdown("### 🎯 Risk Category")
                    emoji = get_category_emoji(category)
                    color = get_category_color(category)
                    
                    st.markdown(
                        f'<div style="background-color: {color}; padding: 2rem; border-radius: 10px; text-align: center;">'
                        f'<h1 style="color: white; margin: 0;">{emoji} {category}</h1>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    st.metric("Risk Classification", category)
                
                with col3:
                    st.markdown("### 💵 Credit Limit")
                    credit_limit = result.get("credit_limit", 0)
                    
                    st.markdown(
                        f'<div style="background-color: #1f77b4; padding: 2rem; border-radius: 10px; text-align: center;">'
                        f'<h1 style="color: white; margin: 0;">{format_currency(credit_limit)}</h1>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    st.metric("Recommended Limit", format_currency(credit_limit))
                
                # Additional metrics (if available)
                st.markdown("---")
                st.markdown("### 📈 Additional Financial Metrics")
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    interest_rate = result.get("interest_rate", 0)
                    st.metric("Interest Rate", format_percentage(interest_rate / 100) if interest_rate > 1 else format_percentage(interest_rate))
                
                with metric_col2:
                    expected_loss = result.get("expected_loss", 0)
                    st.metric("Expected Loss", format_currency(expected_loss))
                
                with metric_col3:
                    st.metric("Model Accuracy", "89%")
                
                # Expandable section for API details
                st.markdown("---")
                with st.expander("🔍 View API Request/Response Details"):
                    col_req, col_res = st.columns(2)
                    
                    with col_req:
                        st.markdown("**Request Payload:**")
                        st.json(features)
                    
                    with col_res:
                        st.markdown("**API Response:**")
                        st.json(result)
    
    elif "last_result" in st.session_state:
        # Display last result if exists
        st.info("ℹ️ Showing last assessment. Adjust parameters and click 'Assess Credit Risk' for new analysis.")
        
        result = st.session_state.last_result
        
        # Display results in columns (same as above)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📊 Risk Score")
            risk_score = result.get("risk_score", result.get("pd", 0))
            category = get_risk_category(risk_score)
            color = get_category_color(category)
            
            st.markdown(
                f'<div style="background-color: {color}; padding: 2rem; border-radius: 10px; text-align: center;">'
                f'<h1 style="color: white; margin: 0;">{format_percentage(risk_score)}</h1>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.metric("Probability of Default", format_percentage(risk_score))
        
        with col2:
            st.markdown("### 🎯 Risk Category")
            emoji = get_category_emoji(category)
            color = get_category_color(category)
            
            st.markdown(
                f'<div style="background-color: {color}; padding: 2rem; border-radius: 10px; text-align: center;">'
                f'<h1 style="color: white; margin: 0;">{emoji} {category}</h1>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.metric("Risk Classification", category)
        
        with col3:
            st.markdown("### 💵 Credit Limit")
            credit_limit = result.get("credit_limit", 0)
            
            st.markdown(
                f'<div style="background-color: #1f77b4; padding: 2rem; border-radius: 10px; text-align: center;">'
                f'<h1 style="color: white; margin: 0;">{format_currency(credit_limit)}</h1>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.metric("Recommended Limit", format_currency(credit_limit))
        
        # Additional metrics
        st.markdown("---")
        st.markdown("### 📈 Additional Financial Metrics")
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            interest_rate = result.get("interest_rate", 0)
            st.metric("Interest Rate", format_percentage(interest_rate / 100) if interest_rate > 1 else format_percentage(interest_rate))
        
        with metric_col2:
            expected_loss = result.get("expected_loss", 0)
            st.metric("Expected Loss", format_currency(expected_loss))
        
        with metric_col3:
            st.metric("Model Accuracy", "89%")
    
    else:
        # Welcome message when no results yet
        st.info("👈 Enter customer information in the sidebar and click 'Assess Credit Risk' to begin.")
        
        # Show feature descriptions
        st.markdown("### 📘 Model Features Overview")
        
        feature_col1, feature_col2 = st.columns(2)
        
        with feature_col1:
            st.markdown("""
            **Transaction Behavior:**
            - **Recency**: Days since last transaction (lower = more active)
            - **Frequency**: Monthly transaction count (higher = more engaged)
            """)
        
        with feature_col2:
            st.markdown("""
            **Financial Patterns:**
            - **Volatility**: Transaction amount consistency
            - **Average Amount**: Typical transaction size
            - **Weekend Ratio**: Weekend vs weekday activity
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div class="footer">'
        '⚠️ <strong>MVP Version</strong> - Under active development<br>'
        'Full analytics dashboard with historical trends, batch processing, and advanced features coming soon<br>'
        'Model: Random Forest Classifier | Accuracy: 89% | Features: 5 | Training Data: Transaction History'
        '</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
