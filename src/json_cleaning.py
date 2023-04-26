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


def json_cleaning(input_df, dict_columns):
    # Make a copy of the input DataFrame to avoid modifying the original one
    df = input_df.copy()

    # List of column names containing dictionaries and/or JSON strings

    """dict_columns = [
        "creator",
        "category",
        "location",
        "photo",
        "profile",
        "urls",
    ]"""

    # Convert JSON strings to dictionaries for each column in dict_columns
    for column in dict_columns:
        df[column] = df[column].apply(
            lambda x: safe_string_to_dict(x) if isinstance(x, str) else x
        )

    # Define the keys to be extracted for each JSON-like column
    all_json_columns = {
        # "creator": [
        # "name",
        # "chosen_currency",
        # "slug",
        # "urls",
        # "is_registered",
        # "id",
        # "avatar",
        # ],
        "category": [
            # "parent_id",
            # "id",
            # "position",
            "name",
            "slug"  # ,
            # "urls",
            # "color"
        ],
        "location": [
            # "short_name",
            # "displayable_name",
            "state"  # ,
            # "id",
            # "name",
            # "slug",
            # "is_root",
            # "localized_name",
            # "type",
            # "country",
            # "urls",
        ],
        # "photo": [
        # "med",
        # "1024x576",
        # "full",
        # "thumb",
        # "key",
        # "small",
        # "little",
        # "1536x864",
        # "ed",
        # ],
        # "profile": [
        # "text_color",
        # "link_text",
        # "background_color",
        # "state_changed_at",
        # "link_text_color",
        # "state",
        # "id",
        # "should_show_feature_image_section",
        # "background_image_opacity",
        # "name",
        # "background_image_attributes",
        # "link_background_color",
        # "feature_image_attributes",
        # "blurb",
        # "project_id",
        # "show_feature_image",
        # "link_url",
        # ],
        # "urls": ["api", "web"],
    }

    # Filter the json_columns dictionary to only include the keys for columns_to_process
    json_columns = {
        column: all_json_columns[column]
        for column in dict_columns
        if column in all_json_columns
    }

    # Extract the keys for each JSON-like column and create new columns
    for column_name, keys in json_columns.items():
        df = extract_dict_columns(df, column_name, keys)

    """
    # List of id columns to be converted to the int data type
    id_columns = [
        "category_id",
        "location_id",
        "profile_id",
    ]

    # Convert the id columns to the int data type
    for column in id_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce", downcast="integer")
        df[column] = df[column].fillna(-1).astype(int)
    """

    """
    # Convert the 'urls_web' column JSON strings to dictionaries if needed
    df["urls_web"] = df["urls_web"].apply(
        lambda x: safe_string_to_dict(x) if isinstance(x, str) else x
    )

    # Define the keys to be extracted from the 'urls_web' column
    url_web_keys = ["project"]

    # Extract the keys from the 'urls_web' column and create new columns
    df = extract_dict_columns(df, "urls_web", url_web_keys)

    # Add urls_web to the dict_columns list
    dict_columns.append("urls_web")
    """

    # Drop the original columns with dictionaries and/or JSON
    df = df.drop(columns=dict_columns)

    # Return the processed DataFrame

    return df
