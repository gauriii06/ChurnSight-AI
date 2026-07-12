import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import shap
import pandas as pd

from models.random_forest_model import model
from preprocessing.preprocessing import X_train, X_test

# Create SHAP explainer for tree models
explainer = shap.TreeExplainer(model)

# Compute SHAP values
shap_values = explainer.shap_values(X_test)

# Global feature importance plot
shap.summary_plot(shap_values, X_test)