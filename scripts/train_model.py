import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Paths
train_path = Path("data/processed/train.csv")
model_dir = Path("models")
model_path = model_dir / "customer_support_model.pkl"

# Create model directory
model_dir.mkdir(parents=True, exist_ok=True)

# Load training data
df = pd.read_csv(train_path)

print("Training records:", len(df))
print("Categories:", df["category"].nunique())

# Create model pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=50000
    )),
    ("classifier", LogisticRegression(
        max_iter=1000
    ))
])

# Train model
print("\nTraining model...")
model.fit(df["text"], df["category"])

# Save model
joblib.dump(model, model_path)

print("\n===== TRAINING COMPLETE =====")
print("Model saved to:", model_path)