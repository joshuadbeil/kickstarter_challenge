import pandas as pd


def drop_duplicates(
    dataframe,
    duplicate_column="id",
    feature_column="usd_type",
    feature_value="domestic",
):
    # Identify the duplicated rows based on the "id" column
    duplicate_mask = dataframe.duplicated("id")

    # Select the rows that are not duplicates and do not have "domestic" as the value in the "usd_type" column
    dataframe = dataframe[
        ~((duplicate_mask) & (dataframe[feature_column] == feature_value))
    ]

    # Drop the rows that are duplicates based on the "id" column, keeping only the first occurrence of each group
    dataframe.drop_duplicates(duplicate_column, keep="first", inplace=True)

    # Return the resulting DataFrame
    return dataframe
