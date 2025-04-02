import pandas as pd
import json
from langchain_core.prompts import ChatPromptTemplate
import os
import random
from dotenv import load_dotenv
from preprocess import store, load_source_file, preprocess_data
load_dotenv()
import re
from collections import defaultdict

from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    deployment_name="gpt-4o",
    model_name="gpt-4o",  # or "gpt-4"
    azure_endpoint="https://triplei-openai.openai.azure.com/",
    openai_api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    openai_api_version="2024-12-01-preview",
    openai_api_type="azure"
)

def describe_dct(df_dct:dict) -> dict:
    sample_dict = {
        col: random.sample(values, min(5, len(values)))
        for col, values in df_dct.items()
    }
    prompt_template = ChatPromptTemplate.from_template(
        """
        You are given a dictionary where each key is a column name and its value contains:
        - The column name again (redundant for clarity)
        - A short list of example values
        
        Your task is to describe what each column represents in plain English (one sentence per column).
        
        Return a valid JSON object with the following format:
        parentheses: 
          "ColumnName1": "Short description of what ColumnName1 represents",
          "ColumnName2": "Short description of what ColumnName2 represents",
          ...
        closed parentheses: 
        
        Only include column names as keys and their descriptions as string values. Do not include the example values in your output.
        
        Input:
        {column_examples}

        """
    )

    prompt = prompt_template.format_messages(
            column_examples= json.dumps(sample_dict, indent=2)

        )

    try:
        response = llm(prompt)
        response_text = response.content.strip()

        # Try to isolate valid JSON from GPT output
        json_start = response_text.find("{")
        if json_start != -1:
            json_data = response_text[json_start:]
            return json.loads(json_data)
        else:
            raise ValueError("No valid JSON found in GPT response")

    except Exception as e:
        print("❌ Error parsing GPT response:", e)
        print("🔍 GPT raw output:\n", response_text if 'response_text' in locals() else "No response")
        return {}

def normalize_schema_dict(data_dict: dict, column_role_map: dict) -> dict:
    """
    Normalize a dictionary of lists using a role mapping like:
    {"OriginalCol": "column_name", ...}
    Only keeps recognized roles, renames keys.
    """
    return {
        new_key: data_dict[old_key]
        for old_key, new_key in column_role_map.items()
        if new_key != "ignore" and old_key in data_dict
    }

def build_column_descriptions_from_dict(normalized_dict: dict) -> dict:
    table_names = normalized_dict.get("table_name", [])
    col_names = normalized_dict.get("column_name", [])
    data_types = normalized_dict.get("data_type", [])

    descriptions = {}

    for table, col, dtype in zip(table_names, col_names, data_types):
        descriptions[col] = f"{dtype} column in {table}"

    return descriptions


def generate_column_mapping_with_openai(df, destination_schema_description):
    """
    Uses OpenAI to map source DataFrame columns to destination schema tables and columns.
    Returns a mapping dictionary: {source_column: {table: ..., column: ...}}
    """
    # with open("prompts/column_mapping.txt", "r", encoding="utf-8") as f:
    #     prompt = ChatPromptTemplate.from_template(f.read())

    prompt = f"""
    You are a data integration assistant. Your task is to help map the columns from a source dataset into a fixed destination schema for sustainability reporting.

    Destination schema (summarized):
    {destination_schema_description}

    Below are the source dataset column names and sample values. Please suggest the most appropriate destination table and column for each source column.

    Format your response as:
    column_name -> table_name.column_name

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
            table, column = target.strip().split(".", 1)

            mapping[source] = {"table": table.strip(), "column": column.strip()}

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
        if "not applicable" in table.lower():
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



if __name__ == "__main__":

        df = load_source_file("cache/translated_df.csv")
        df_dict = preprocess_data(df)
        destination_tables = load_source_file('data/DestinationTables.xlsx')
        table_examples_text = extract_table_examples(destination_tables, max_rows=2)
        with open("cache/table_examples.txt", "w", encoding="utf-8") as f:
            f.write(table_examples_text)

        with open('cache/table_examples.txt', 'r') as f:
            table_examples = f.read()

        print(table_examples)


        with open('prompts/schema_description.txt', 'r', encoding="utf-8") as f:
            schema_description = f.read()
        # mappings = generate_column_mapping_with_openai(df_dict, schema_description)
        # store(mappings, 'cache/mapping.json')
        with open('cache/mapping.json', 'r', encoding="utf-8") as f:
            mapping = json.load(f)
        # print(mapping)

        # clean_mapping = clean_mapping_dict(mapping)
        # store(clean_mapping, 'cache/clean_mapping.json')
        with open('cache/clean_mapping.json', 'r', encoding="utf-8") as f:
            clean_mapping = json.load(f)
        # print(clean_mapping)
        #
        # destination_dfs = split_dataframe_by_mapping(df, clean_mapping)
        #
        # export_to_excel_with_sheets(destination_dfs, "data/output/mapped_output.xlsx")
        #
        # print(destination_dfs)
        # schema = pd.read_excel("data/source/DestinationSchema.xlsx")
        # df_dct = df.to_dict(orient="list")
        # schema = schema.dropna(axis=1)
        # schema_dct = schema.to_dict(orient="list")

        # source_descriptions = describe_dct(df_dct)
        # store(source_descriptions, 'cache/source_descriptions.json')
        # with open('cache/source_descriptions.json', 'r', encoding="utf-8") as f:
        #     source_descriptions = json.load(f)
        # print(source_descriptions)
        # schema_descriptions = describe_dct(schema_dct)
        # store(schema_descriptions, 'cache/schema_descriptions.json')
        # with open('cache/schema_descriptions.json', 'r', encoding="utf-8") as f:
        #     schema_descriptions = json.load(f)
        # col_mappings = generate_schema_column_roles(schema_dct)
        # store(col_mappings, 'cache/col_mappings.json')
        # with open('cache/col_mappings.json', 'r', encoding="utf-8") as f:
        #     col_mappings = json.load(f)
        # print(col_mappings)
        # normalized_df = normalize_schema_dict(schema_dct, col_mappings)
        # col_descr = build_column_descriptions_from_dict(normalized_df)
        # print(col_descr)
        # mapping = map_columns_to_schema(source_descriptions, col_descr)
        # print(mapping)