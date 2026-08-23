import pandas as pd
import re
from pathlib import Path

# Paths
input_path = Path("data/raw/train.csv")
output_dir = Path("data/processed")
output_path = output_dir / "cleaned_train.csv"

# Create output folder if needed
output_dir.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(input_path)

print("Original shape:", df.shape)

# Clean text
df["text"] = df["text"].astype(str)
df["text"] = df["text"].apply(lambda x: re.sub(r"\s+", " ", x).strip())

# Remove exact duplicates
df = df.drop_duplicates()

# Remove rows with empty text/category
df = df.dropna(subset=["text", "category"])
df = df[df["text"].str.strip() != ""]

# Save cleaned dataset
df.to_csv(output_path, index=False)

print("Cleaned shape:", df.shape)
print("Saved to:", output_path)
print("Cleaning completed successfully!")