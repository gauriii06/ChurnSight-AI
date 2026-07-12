import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("data/telco_churn.csv")

# Remove customerID
data.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

# Fill missing values
data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())

# Convert churn to binary
data["Churn"] = data["Churn"].map({"Yes":1, "No":0})

# Identify categorical columns
categorical_cols = data.select_dtypes(include=["object","string"]).columns

# One-hot encode categorical features
data = pd.get_dummies(data, columns=categorical_cols)

# Feature matrix and target vector
X = data.drop("Churn", axis=1)
y = data["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)