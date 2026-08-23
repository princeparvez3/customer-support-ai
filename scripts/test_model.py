import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

model_path = Path("models/customer_support_model.pkl")
test_path = Path("data/processed/test.csv")

model = joblib.load(model_path)
df = pd.read_csv(test_path)

predictions = model.predict(df["text"])

accuracy = accuracy_score(df["category"], predictions)

print("===== FINAL TEST EVALUATION =====")
print(f"Test records: {len(df)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy percentage: {accuracy * 100:.2f}%")

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        df["category"],
        predictions,
        zero_division=0
    )
)