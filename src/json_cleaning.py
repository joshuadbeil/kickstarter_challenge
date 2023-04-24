import pandas as pd
import json


def safe_string_to_dict(s):
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return s


def extract_dict_columns(df, column, keys):
    for key in keys:
        df[f"{column}_{key}"] = df[column].apply(
            lambda x: x.get(key) if isinstance(x, dict) else None
        )
    return df


def process_dataframe(input_df):
    # Make a copy of the input DataFrame to avoid modifying the original one
    df = input_df.copy()

    # List of column names containing dictionaries and/or JSON strings
    dict_columns = [
        "category",
    ]

    # Convert JSON strings to dictionaries for each column in dict_columns
    for column in dict_columns:
        df[column] = df[column].apply(
            lambda x: safe_string_to_dict(x) if isinstance(x, str) else x
        )

    # Define the keys to be extracted for each JSON-like column
    json_columns = {
        "category": ["id", "name", "slug"],
    }

    # Extract the keys for each JSON-like column and create new columns
    for column_name, keys in json_columns.items():
        df = extract_dict_columns(df, column_name, keys)

    # List of id columns to be converted to the int data type
    id_columns = [
        "category_id",
    ]

    # Convert the id columns to the int data type
    for column in id_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce", downcast="integer")
        df[column] = df[column].fillna(-1).astype(int)

    # Drop the original columns with dictionaries and/or JSON
    df = df.drop(columns=dict_columns)

    # Return the processed DataFrame

    return df
