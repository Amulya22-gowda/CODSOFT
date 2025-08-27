import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# --------------------------
# Step 1 — Load Dataset
# --------------------------
train_df = pd.read_csv("fraudTrain.csv")
test_df = pd.read_csv("fraudTest.csv")
df = pd.concat([train_df, test_df], ignore_index=True)

print("✅ Dataset loaded. Shape:", df.shape)

# --------------------------
# Step 2 — Drop irrelevant / high-cardinality columns
# --------------------------
drop_cols = [
    'Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'first', 'last',
    'street', 'city', 'state', 'zip', 'dob', 'trans_num',
    'merchant', 'job'  # high-cardinality categorical columns
]
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# --------------------------
# Step 3 — One-hot encode remaining categorical columns
# --------------------------
df = pd.get_dummies(df, drop_first=True)

# --------------------------
# Step 4 — Separate features & target
# --------------------------
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# --------------------------
# Step 5 — Scale features
# --------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------
# Step 6 — Train-Test Split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------
# Step 7 — Train Model
# --------------------------
model = LogisticRegression(max_iter=500, class_weight='balanced') # more iterations for convergence
model.fit(X_train, y_train)

# --------------------------
# Step 8 — Evaluate Model
# --------------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

from sklearn.metrics import precision_score, recall_score, f1_score

precision_fraud = precision_score(y_test, y_pred, pos_label=1)
recall_fraud = recall_score(y_test, y_pred, pos_label=1)
f1_fraud = f1_score(y_test, y_pred, pos_label=1)

print(f"Fraud Precision: {precision_fraud:.4f}")
print(f"Fraud Recall: {recall_fraud:.4f}")
print(f"Fraud F1-Score: {f1_fraud:.4f}")

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# --------------------------
# Step 9 — Save model & scaler
# --------------------------
joblib.dump(model, "credit_card_fraud_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")

print("✅ Model and scaler saved successfully!")


