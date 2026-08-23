import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Paths
input_path = Path("data/processed/cleaned_train.csv")
output_dir = Path("data/processed")

# Load cleaned dataset
df = pd.read_csv(input_path)

print("Total records:", len(df))
print("Total categories:", df["category"].nunique())

# 80% train, 20% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["category"]
)

# Split remaining 20% into 10% validation + 10% test
validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["category"]
)

# Save
train_df.to_csv(output_dir / "train.csv", index=False)
validation_df.to_csv(output_dir / "validation.csv", index=False)
test_df.to_csv(output_dir / "test.csv", index=False)

print("\n===== SPLIT COMPLETE =====")
print("Train:", len(train_df))
print("Validation:", len(validation_df))
print("Test:", len(test_df))