import streamlit as st
import joblib
import pandas as pd

# Load model, scaler, and feature names
model = joblib.load("credit_card_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("💳 Credit Card Fraud Detection")

st.write("Fill in the transaction details below:")

# Create a dictionary for user input
user_input = {}

for col in feature_names:
    if col.startswith("category_"):
        user_input[col] = st.checkbox(col.replace("category_", "").replace("_", " ").title())
    elif col.startswith("gender_"):
        user_input[col] = st.checkbox(col.replace("gender_", "").title())
    else:
        user_input[col] = st.number_input(f"{col}", value=0.0)

# Convert checkboxes (True/False) to 1/0
for col in feature_names:
    if isinstance(user_input[col], bool):
        user_input[col] = int(user_input[col])

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Scale numeric values
scaled_input = scaler.transform(input_df)

# Prediction
if st.button("Predict Fraud"):
    prediction = model.predict(scaled_input)[0]
    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")
