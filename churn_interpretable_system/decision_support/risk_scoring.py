import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.random_forest_model import model
from preprocessing.preprocessing import X_test

# Get churn probabilities
probabilities = model.predict_proba(X_test)[:,1]

# Risk classification function
def classify_risk(prob):
    
    if prob < 0.3:
        return "LOW"
    
    elif prob < 0.6:
        return "MEDIUM"
    
    else:
        return "HIGH"

# Generate risk scores
risk_scores = [classify_risk(p) for p in probabilities]

# Print sample results
for i in range(10):
    print(f"Customer {i}")
    print("Churn Probability:", round(probabilities[i],3))
    print("Risk Level:", risk_scores[i])
    print()