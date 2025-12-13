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