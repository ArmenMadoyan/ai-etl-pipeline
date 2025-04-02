import pandas as pd

# Load the source data
source_path = "data/source/SourceData.csv"  # or use .csv if applicable
df = pd.read_csv(source_path)  # or pd.read_csv(source_path)

# Basic inspection
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])
print("\nColumn names:\n", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nMissing values per column:\n", df.isnull().sum())
print("\nFirst 5 rows:\n", df.head())
