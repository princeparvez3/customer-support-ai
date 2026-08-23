import pandas as pd
import joblib
from pathlib import Path

model_path = Path("models/customer_support_model.pkl")
validation_path = Path("data/processed/validation.csv")
output_path = Path("reports/validation_predictions.csv")

model = joblib.load(model_path)
df = pd.read_csv(validation_path)

predictions = model.predict(df["text"])

df["predicted_category"] = predictions
df["correct"] = df["category"] == df["predicted_category"]

errors = df[df["correct"] == False].copy()

errors = errors[
    ["text", "category", "predicted_category"]
]

errors.to_csv(output_path, index=False)

print("===== ERROR ANALYSIS =====")
print("Validation records:", len(df))
print("Incorrect predictions:", len(errors))
print("Correct predictions:", len(df) - len(errors))
print("Error rate:", f"{len(errors) / len(df) * 100:.2f}%")
print("Saved to:", output_path)

print("\n===== SAMPLE ERRORS =====")
print(errors.head(10).to_string(index=False))