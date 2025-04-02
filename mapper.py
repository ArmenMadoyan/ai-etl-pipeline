import pandas as pd
import json
import config
from dotenv import load_dotenv
from preprocess import store, load_source_file, preprocess_data
load_dotenv()
import re
from collections import defaultdict

from langchain_openai import AzureChatOpenAI


def generate_column_mapping_with_openai(df, destination_schema_description, table_column_map):
    """
    Uses OpenAI to map source DataFrame columns to destination schema tables and columns.
    Returns a mapping dictionary: {source_column: {table: ..., column: ...}}
    """


    prompt = f"""
    You are a data integration assistant. Your task is to help map the columns from a source dataset into a fixed destination schema for reporting.

    Destination schema (summarized):
    {destination_schema_description}
    
    Here are the available destination tables and their columns:
    """
    for table, columns in table_column_map.items():
        prompt += f"\n{table}: {', '.join(columns)}"

    prompt += """
    Below are the source dataset column names and sample values. Please suggest the most appropriate destination table and column for each source column.

    Format your response using **only this format**, one per line:

    column_name -> table_name.column_name
    
    For uncertain or irrelevant columns:
    - If you're unsure or need manual review:  
      column_name -> Unclear (needs review)
    
    ⚠️ Do not use bullet points, Markdown, quotes, or formatting. Just plain mappings.

    Source columns:
    """

    for col in df.columns:
        sample_values = df[col].dropna().astype(str).unique()[:5].tolist()
        prompt += f"\n- {col}: {sample_values}"

    messages=[{"role": "user", "content": prompt}]

    # Call the LLM
    print("🧠 Invoking LLM for schema mapping...")
    result = llm.invoke(messages)
    raw_text = result.content
    print("✅ LLM Response:\n", raw_text)

    # Parse mapping from response
    mapping = {}
    for line in raw_text.strip().splitlines():
        if "->" in line:
            source, target = line.split("->")
            source = source.strip()
            target = target.strip()

            if "." in target:
                table, column = target.split(".", 1)
                mapping[source] = {"table": table.strip(), "column": column.strip()}
            else:
                mapping[source] = {"table": target, "column": ""}

    return mapping


def clean_mapping_dict(raw_mapping_dict):
    """
    Cleans a raw OpenAI/LangChain-generated mapping dictionary into a usable structure.
    - Strips numbers, bold markers, and extra description.
    - Filters out entries that are marked 'Not applicable'.
    """
    cleaned = {}

    for raw_key, value in raw_mapping_dict.items():
        # Clean key like "1. **merchant**" → "merchant"
        clean_key = re.sub(r"^\d+\.\s*\*\*|\*\*", "", raw_key).strip().lower()

        table = value["table"].strip()
        column = value["column"].strip()

        # Skip non-applicable rows
        if "unclear" in table.lower():
            continue

        # Remove extra explanation in parentheses from column
        column = re.sub(r"\s*\(.*?\)", "", column).strip()

        cleaned[clean_key] = {
            "table": table,
            "column": column
        }

    return cleaned


def split_dataframe_by_mapping(df, cleaned_mapping):
    """
    Splits the source DataFrame into destination tables based on mapping.
    Returns a dict: {table_name: DataFrame}
    """
    table_dfs = defaultdict(dict)

    for source_col, mapping in cleaned_mapping.items():
        table = mapping["table"]
        dest_col = mapping["column"]

        if source_col in df.columns and table and dest_col:
            table_dfs[table][dest_col] = df[source_col]

    # Convert column dicts to DataFrames
    return {table: pd.DataFrame(data) for table, data in table_dfs.items()}

def export_to_excel_with_sheets(table_dfs, output_path):
    """
    Exports multiple DataFrames to an Excel file with each DataFrame in its own sheet.

    Args:
        table_dfs: dict of {table_name: DataFrame}
        output_path: file path for the Excel file (e.g., "output_tables.xlsx")
    """
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        for table_name, df in table_dfs.items():
            # Truncate sheet name if it's too long for Excel
            sheet_name = table_name[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"✅ Excel file saved to: {output_path}")


def extract_table_examples(data, max_rows=2):
    """
    Extracts sample rows from a DataFrame or dict of DataFrames and returns JSON-style structure:
    {TABLE_NAME: {COLUMN_NAME: [VALUES]}}

    Args:
        data: pd.DataFrame or dict of {sheet_name: pd.DataFrame}
        max_rows: Number of rows to extract per table

    Returns:
        dict: {table_name: {column_name: [values]}}
    """
    result = {}

    if isinstance(data, dict):  # Excel with multiple sheets
        for sheet_name, df in data.items():
            if df.empty:
                continue
            table_data = df.head(max_rows).to_dict(orient="list")
            result[sheet_name.strip()] = table_data

    elif isinstance(data, pd.DataFrame):  # Single CSV
        result["SourceData"] = data.head(max_rows).to_dict(orient="list")

    else:
        raise ValueError("Unsupported data type. Must be DataFrame or dict of DataFrames.")

    return result

def extract_table_structure(data):
    """
    Extracts only table names and column names.

    Args:
        data: pd.DataFrame or dict of {sheet_name: pd.DataFrame}

    Returns:
        dict: {table_name: [column_name, ...]}
    """
    result = {}

    if isinstance(data, dict):  # Excel with multiple sheets
        for sheet_name, df in data.items():
            if df.empty:
                continue
            result[sheet_name.strip()] = df.columns.astype(str).tolist()

    elif isinstance(data, pd.DataFrame):  # Single CSV
        result["SourceData"] = data.columns.astype(str).tolist()

    else:
        raise ValueError("Unsupported data type. Must be DataFrame or dict of DataFrames.")

    return result


if __name__ == "__main__":
    llm = AzureChatOpenAI(
        deployment_name=config.AZURE_OPENAI_DEPLOYMENT,
        model_name=config.AZURE_OPENAI_MODEL,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        openai_api_key=config.AZURE_OPENAI_KEY,
        openai_api_version=config.AZURE_OPENAI_VERSION,
        openai_api_type=config.AZURE_OPENAI_TYPE,
        temperature=0
    )

    df = load_source_file(config.TRANSLATED_MAP_OUTPUT_PATH)
    df, logs = preprocess_data(df)



    destination_tables_example = load_source_file(config.DESTINATION_TABLES_PATH)

    table_columns = extract_table_structure(destination_tables_example)
    store(table_columns, 'cache/table_columns.json')

    with open('cache/table_columns.json', 'r', encoding="utf-8") as f:
        table_columns = json.load(f)

    with open(config.PROMPT_SCHEMA_DESCRIPTION, 'r', encoding="utf-8") as f:
        schema_description = f.read()

    mappings = generate_column_mapping_with_openai(df, schema_description, table_columns)
    store(mappings, 'cache/mapping.json')
    with open('cache/mapping.json', 'r', encoding="utf-8") as f:
        mapping = json.load(f)

    clean_mapping = clean_mapping_dict(mapping)
    store(clean_mapping, 'cache/clean_mapping.json')

    with open('cache/clean_mapping.json', 'r', encoding="utf-8") as f:
        clean_mapping = json.load(f)

    destination_dfs = split_dataframe_by_mapping(df, clean_mapping)
    print(clean_mapping)
    export_to_excel_with_sheets(destination_dfs, config.MAPPED_OUTPUT_PATH)
