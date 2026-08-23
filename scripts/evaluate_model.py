import pandas as pd
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Paths
model_path = Path("models/customer_support_model.pkl")
validation_path = Path("data/processed/validation.csv")

# Load model and validation data
model = joblib.load(model_path)
df = pd.read_csv(validation_path)

# Predictions
predictions = model.predict(df["text"])

# Accuracy
accuracy = accuracy_score(df["category"], predictions)

print("===== MODEL EVALUATION =====")
print(f"Validation records: {len(df)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy percentage: {accuracy * 100:.2f}%")

# Detailed report
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(
    df["category"],
    predictions,
    zero_division=0
))