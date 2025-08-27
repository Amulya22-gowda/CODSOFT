import pandas as pd
import joblib

# Load dataset and feature names
df = pd.read_csv("fraudTest.csv")
feature_names = joblib.load("feature_names.pkl")

# === Apply same preprocessing as in training ===
# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['category', 'gender'], drop_first=False)

# Keep only the feature columns used for training
df = df[feature_names]

# Get one legit and one fraud row
# (These rows now match the exact columns the model expects)
original_df = pd.read_csv("fraudTest.csv")  # for is_fraud filtering
legit_index = original_df[original_df['is_fraud'] == 0].index[0]
fraud_index = original_df[original_df['is_fraud'] == 1].index[0]

legit_row = df.iloc[legit_index]
fraud_row = df.iloc[fraud_index]

def print_values(row, label):
    print(f"\n--- {label} Example ---")
    for val in row:
        print(val)

print_values(legit_row, "Legitimate Transaction")
print_values(fraud_row, "Fraudulent Transaction")
