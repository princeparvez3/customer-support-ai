import pandas as pd

# Load raw dataset
df = pd.read_csv("data/raw/train.csv")

print("===== DATASET INSPECTION =====")

# Number of records and columns
print("\n1. Dataset Shape:")
print(df.shape)

# Column names
print("\n2. Columns:")
print(df.columns.tolist())

# Empty values
print("\n3. Empty Values:")
print(df.isnull().sum())

# Number of unique categories
print("\n4. Number of Unique Categories:")
print(df["category"].nunique())

# Category names
print("\n5. Categories:")
print(df["category"].unique())

# Duplicate rows
print("\n6. Duplicate Rows:")
print(df.duplicated().sum())

# First 5 records
print("\n7. First 5 Records:")
print(df.head())