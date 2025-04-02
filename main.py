import pandas as pd
import json
import config
from preprocess import load_source_file, preprocess_data, store
# Load the source data
source_path = config.SOURCE_FILE_PATH
df = load_source_file(source_path)
df,_ = preprocess_data(df)

