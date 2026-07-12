import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.random_forest_model import model
from preprocessing.preprocessing import X_test
from decision_support.risk_scoring import classify_risk

import shap

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Compute SHAP values
shap_values = explainer.shap_values(X_test)

# Get churn probabilities
probabilities = model.predict_proba(X_test)[:,1]

# Recommendation rules
def generate_recommendations(features):

    actions = []

    if "Contract_Month-to-month" in features:
        actions.append("Offer discounted yearly subscription")

    if "MonthlyCharges" in features:
        actions.append("Provide loyalty discount")

    if "tenure" in features:
        actions.append("Provide onboarding engagement offers")

    if "TechSupport_No" in features:
        actions.append("Promote tech support service")

    return actions
# Generate recommendations for first 5 customers
for i in range(5):

    prob = probabilities[i]
    risk = classify_risk(prob)

    # Get SHAP values for customer
    shap_contributions = shap_values[1][i]

    # Identify top contributing features
    top_features = X_test.columns[abs(shap_contributions).argsort()[-3:]]

    recommendations = generate_recommendations(top_features)

    print(f"Customer {i}")
    print("Churn Probability:", round(prob,3))
    print("Risk Level:", risk)
    print("Key Drivers:", list(top_features))
    print("Recommended Actions:", recommendations)
    print()