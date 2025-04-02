import pandas as pd
import random
import re
import json
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict
import os
from dotenv import load_dotenv
import requests
load_dotenv()
import time

from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    deployment_name="gpt-4o",
    model_name="gpt-4o",  # or "gpt-4"
    azure_endpoint="https://triplei-openai.openai.azure.com/",
    openai_api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    openai_api_version="2024-12-01-preview",
    openai_api_type="azure"
)


def extract_unique_text_values(df, log_interval=1):
    """
    Extract unique non-null string values from all text (object) columns.
    Logs progress for each column including number of duplicates removed.
    Returns a dictionary: {column_name: list of unique values}.
    """
    unique_values = {}
    text_columns = df.select_dtypes(include=["object"]).columns
    print(f"🔍 Found {len(text_columns)} text columns to process...\n")

    for idx, col in enumerate(text_columns, 1):
        print(f"➡️  [{idx}/{len(text_columns)}] Processing column: '{col}'")
        cleaned = df[col].dropna().astype(str).str.strip()
        total = len(cleaned)
        uniques = cleaned.unique().tolist()
        duplicates_removed = total - len(uniques)

        print(f"   ↪️  Found {len(uniques)} unique values in '{col}'")
        print(f"   🗑️  Removed {duplicates_removed} duplicates in '{col}'\n")

        unique_values[col] = uniques

    print("✅ Done extracting unique values from all text columns.\n")
    return unique_values

def classify_text_columns(columns: Dict[str, List[str]]) -> Dict[str, str]:
    # Template prompt for classification
    with open("prompts/text_column_classifier.txt", "r", encoding="utf-8") as f:
        prompt_template = ChatPromptTemplate.from_template(f.read())

    results = {}
    for col, values in columns.items():
        top_ = values[:10] if len(values) >= 10 else values
        prompt = prompt_template.format_messages(
            column_name=col,
            sample_values="\n- " + "\n- ".join(top_)
        )
        try:
            response = llm.invoke(prompt)
            classification = response.content.strip().upper()
            results[col] = classification if classification in ["TEXT", "NON-TEXT"] else "UNKNOWN"
        except Exception as e:
            print(f"Error classifying column '{col}': {e}")
            results[col] = "ERROR"
    return results

def detect_non_english_columns(columns: Dict[str, List[str]]) -> Dict[str, str]:
    # Template prompt for classification
    with open("prompts/detect_language.txt", "r", encoding="utf-8") as f:
        prompt_template = ChatPromptTemplate.from_template(f.read())

    results = {}
    for col, values in columns.items():
        sampled_values = random.sample(values, min(len(values), 10))
        prompt = prompt_template.format_messages(
            column_name=col,
            sample_values="\n- " + "\n- ".join(sampled_values)
        )
        try:
            response = llm.invoke(prompt)
            label = response.content.strip().upper()
            results[col] = label if label in ["ENGLISH", "NON-ENGLISH"] else "UNKNOWN"
        except Exception as e:
            print(f"Error processing column '{col}': {e}")
            results[col] = "ERROR"
    return results

def translate_to_english(text_json, subscription_key, endpoint, region, batch_size=100):
    """
    Translates a dictionary where each key maps to a list of non-English values.
    Uses batching for efficient translation via Azure Translator API.
    Returns a dictionary with the same keys and translated value lists.
    """
    path = "/translate?api-version=3.0&to=en"
    constructed_url = endpoint + path
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Ocp-Apim-Subscription-Region': region,
        'Content-type': 'application/json'
    }

    translation_map_per_column = {}

    for col, values in text_json.items():
        print(f"\n🌍 Translating column '{col}' with {len(values)} unique values...")
        translation_map = {}

        for i in range(0, len(values), batch_size):
            batch = values[i:i + batch_size]
            body = [{'text': str(v)} for v in batch]

            try:
                response = requests.post(constructed_url, headers=headers, json=body)
                if response.status_code == 200:
                    translations = [item['translations'][0]['text'] for item in response.json()]
                    for original, translated in zip(batch, translations):
                        translation_map[original] = translated
                    print(f"   ✅ Translated batch {i + 1}-{i + len(batch)}")
                else:
                    print(f"   ⚠️ Error {response.status_code} for batch {i + 1}-{i + len(batch)}")
                    for original in batch:
                        translation_map[original] = original  # fallback
            except Exception as e:
                print(f"   ❌ Exception: {e}")
                for original in batch:
                    translation_map[original] = original

            time.sleep(0.15)

        translation_map_per_column[col] = translation_map
        print(f"🔹 Done: {col} — {len(translation_map)} values translated.")

    return translation_map_per_column

def text_translate_to_english(text, subscription_key, endpoint, region):
    path = '/translate?api-version=3.0&to=en'
    constructed_url = endpoint + path

    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Ocp-Apim-Subscription-Region': region,
        'Content-type': 'application/json'
    }

    body = [{'text': text}]
    response = requests.post(constructed_url, headers=headers, json=body)


    try:
        data = response.json()
        if isinstance(data, list):
            return data[0]['translations'][0]['text']
        else:
            print(f"❌ Unexpected response format: {data}")
            return text  # fallback: original
    except Exception as e:
        print(f"⚠️ Translation failed for '{text}': {e}")
        print(f"Status: {response.status_code}, Response: {response.text}")
        return text  # fallback

if __name__ == "__main__":
    from preprocess import store, load_source_file, preprocess_data

    source_path = "data/source/SourceData.csv"  # or use .csv if applicable
    df_raw = load_source_file(source_path)

    df, logs = preprocess_data(df_raw)

    # Store Unique values
    # dct_unique = extract_unique_text_values(df)
    # store(dct_unique, 'cache/unique_values.json')
    with open('cache/unique_values.json', 'r') as f:
        unique_values = json.load(f)


    # clasified_columns = classify_text_columns(unique_values)
    # text_columns = {col: label for col, label in clasified_columns.items() if label == "TEXT"}
    #
    # filtered_unique_values = {
    #     col: values
    #     for col, values in unique_values.items()
    #     if col in text_columns
    # }
    # store(filtered_unique_values, 'cache/clasified_columns.json')
    with open("cache/clasified_columns.json", "r") as f:
        filtered_unique_values = json.load(f)
    #
    #
    # non_english = detect_non_english_columns(filtered_unique_values)
    # nonenglish = {col: label for col, label in non_english.items() if label == "NON-ENGLISH"}
    # filtered_non_english = {
    #     col: values
    #     for col, values in unique_values.items()
    #     if col in nonenglish
    # }

    # store(filtered_non_english, "cache/non_english.json")
    with open('cache/non_english.json', 'r', encoding="utf-8") as f:
        non_english = json.load(f)


    subscription_key = os.getenv('AZURE_TRANSLATE_API_KEY')
    endpoint = 'https://api.cognitive.microsofttranslator.com/'
    region = 'eastus'

    # translated = translate_to_english(non_english, subscription_key, endpoint, region)
    # store(translated, "cache/translated-map.json")
    with open('cache/translated-map.json', 'r', encoding="utf-8") as f:
        translation_maps = json.load(f)
    # print(translation_maps.keys())

    df_copy = df.copy()

    for col, value_map in translation_maps.items():
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].map(lambda x: value_map.get(x, x))
            print(f"✅ Translated column: {col}")
        else:
            print(f"⚠️ Column '{col}' not found in DataFrame – skipped.")



    df_copy.to_csv('cache/translated_df.csv', index=False)
