import joblib
import numpy as np

# Load saved model, scaler, and feature names
model = joblib.load("credit_card_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

print("Credit Card Fraud Detection System")
print("Enter transaction details to predict:")

# Get feature values from user
features = []
for col in feature_names:
    val = float(input(f"Enter value for '{col}': "))
    features.append(val)

# Convert to numpy array and reshape
features_array = np.array(features).reshape(1, -1)

# Scale features
features_scaled = scaler.transform(features_array)

# Predict
prediction = model.predict(features_scaled)

if prediction[0] == 1:
    print("🚨 FRAUDULENT transaction detected!")
else:
    print("✅ Legitimate transaction.")
