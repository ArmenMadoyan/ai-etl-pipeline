# config.py
from dotenv import load_dotenv
import os

load_dotenv()

# --- Preprocessing ---
SOURCE_FILE_PATH = "data/source/SourceData.csv"
DESTINATION_TABLES_PATH = "data/source/DestinationTables.xlsx"
TRANSLATED_MAP_OUTPUT_PATH = "data/output/translated_map.csv"
LOG_PREVIEW_COUNT = 10
MAPPED_OUTPUT_PATH = "data/output/mapped_output.xlsx"


# --- Azure Translator Config ---
AZURE_TRANSLATE_KEY = os.getenv("AZURE_TRANSLATE_API_KEY")
AZURE_TRANSLATE_ENDPOINT = "https://api.cognitive.microsofttranslator.com/"
AZURE_TRANSLATE_REGION = "eastus"


# --- Azure OpenAI LLM ---
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = "https://triplei-openai.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT = "gpt-4o"
AZURE_OPENAI_MODEL = "gpt-4o"
AZURE_OPENAI_VERSION = "2024-12-01-preview"
AZURE_OPENAI_TYPE = "azure"

# --- Prompt Templates ---
PROMPT_TEXT_COLUMN_CLASSIFIER = "prompts/text_column_classifier.txt"
PROMPT_LANGUAGE_DETECTION = "prompts/detect_language.txt"
PROMPT_SCHEMA_DESCRIPTION = "prompts/schema_description.txt"
