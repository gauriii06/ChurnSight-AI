import pandas as pd

# Load dataset
data = pd.read_csv("data/telco_churn.csv")

# Display first rows
print("First 5 rows:")
print(data.head())

# Dataset shape
print("\nDataset shape:")
print(data.shape)

# Data information
print("\nDataset info:")
print(data.info())

# Statistical summary
print("\nStatistical summary:")
print(data.describe())

# Missing values
print("\nMissing values:")
print(data.isnull().sum())

# Target distribution
print("\nChurn distribution:")
print(data["Churn"].value_counts())