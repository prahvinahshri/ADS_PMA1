import pandas as pd

try:
    from data_utils import load_data
except ImportError:
    from src.data_utils import load_data

def check_missing_values(df):
    missing = df.isnull().sum()
    return missing[missing > 0]

def check_duplicates(df):
    return int(df.duplicated().sum())

def check_value_range(df, column, min_val, max_val):
    return df[(df[column] < min_val) | (df[column] > max_val)]

if __name__ == "__main__":
    df = load_data()
    print(f"Rows: {len(df):,} | Columns: {df.shape[1]}\n")

    missing = check_missing_values(df)
    print("Missing values:")
    print(missing if not missing.empty else "  None found")

    print(f"\nDuplicate rows: {check_duplicates(df)}")

    invalid = check_value_range(df, "Cleanliness", 1, 5)
    print(f"\nInvalid Cleanliness ratings: {len(invalid)}")
