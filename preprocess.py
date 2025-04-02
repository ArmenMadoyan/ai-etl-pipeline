import pandas as pd
import os
import json
import numpy as np
import config

def load_source_file(filepath):
    """
    Loads a source file which can be either a CSV or an Excel file with single or multiple sheets.

    Returns:
        - DataFrame if CSV
        - Dict of DataFrames if Excel (multi-sheet)
    """
    ext = os.path.splitext(filepath)[-1].lower()

    if ext == '.xlsx':
        data = pd.read_excel(filepath, sheet_name=None)  # load all sheets
        print(f"✅ Loaded Excel file with {len(data)} sheets.")
        return data
    elif ext == '.csv':
        df = pd.read_csv(filepath, low_memory=False)
        print(f"✅ Loaded CSV file with shape: {df.shape}")
        return df
    else:
        raise ValueError(f"❌ Unsupported file format: {ext}")

def preprocess_data(df):
    log = []

    # 1. Normalize column names
    original_cols = df.columns.tolist()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w_]", "", regex=True)
    )
    log.append(f"Normalized column names from: {original_cols} to: {df.columns.tolist()}")

    # 2. Drop fully empty columns
    empty_cols = df.columns[df.isnull().all()].tolist()
    if empty_cols:
        log.append(f"Dropped fully empty columns: {empty_cols}")
        df = df.drop(columns=empty_cols)

    # 3. Drop unnamed columns
    unnamed_cols = [col for col in df.columns if col.startswith("unnamed")]
    if unnamed_cols:
        log.append(f"Dropped 'unnamed' columns: {unnamed_cols}")
        df = df.drop(columns=unnamed_cols)

    # 4. Remove duplicate columns
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        log.append(f"Dropped duplicate columns: {duplicate_cols}")
        df = df.loc[:, ~df.columns.duplicated()]

    # 5. Replace placeholders like 'in', 'n/a', '-', etc.
    placeholders = ["n/a", "none", "-", "null", ""]
    df.replace(placeholders, np.nan, inplace=True)
    log.append("Replaced placeholder strings with NaN.")

    # 6. Strip whitespace from object columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()

    # 7. Attempt numeric conversion
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            continue  # Leave as-is

    # 8. Attempt datetime conversion on suspected date columns
    for col in df.columns:
        if "date" in col or "time" in col:
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                log.append(f"Converted column '{col}' to datetime.")
            except Exception:
                continue

    # 9. Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        log.append(f"Dropped {before - after} duplicate rows.")

    return df, log



def store(new_data: dict, filename: str):
    def convert(obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    cleaned_data = convert(new_data)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    print(f"✅ File '{filename}' written (overwritten if it already existed).")


if __name__ == "__main__":


    source_path = config.DESTINATION_TABLES_PATH
    df = load_source_file(source_path)

    clean_df, logs = preprocess_multiple_sheets(df)
    for entry in logs:
        print("[LOG]", entry)
    print(clean_df.columns)

