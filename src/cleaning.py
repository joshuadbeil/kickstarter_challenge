import os
import pandas as pd
import datetime as dt

import json_cleaning as jc

""" GETTING SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Would like to figure this out in the future (ask GPT 4)
"""
import warnings
warnings.filterwarnings('ignore')


def convert_goal_to_usd(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the funding goal into USD to make it comparable across all projects"""

    df['converted_goal'] = (df['goal'].mul(df['static_usd_rate'])).round(2)

    df = df.drop(['goal','static_usd_rate'], axis =1)
    return df


def convert_backers_pledged(df: pd.DataFrame) -> pd.DataFrame:
    """Create a new column for pledge per backer and drop backers_count afterwards"""

    df['pledge_per_backer'] = (df['usd_pledged'] / df['backers_count']).round(2)
    df['pledge_per_backer'] = df['pledge_per_backer'].fillna(0)
    df['usd_pledged'] = df['usd_pledged'].round(2)

    df = df.drop(['backers_count'], axis =1)
    return df


def convert_string_wordcount(df: pd.DataFrame) -> pd.DataFrame:
    """Create new columns that contain the word counts for project name and blurb columns"""

    df['len_blurb'] = df['blurb'].str.split().str.len()
    df['len_name'] = df['name'].str.split().str.len()

    df = df.drop(['blurb', 'name'], axis = 1)
    return df

def convert_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Decipher the json-format in the category column into category_name and parent_category columns"""

    col = ['category']
    df = jc.json_cleaning(df, col)
    df['parent_category'] = df['category_slug'].apply(lambda x: x.split('/')[0])

    df = df.drop(['category_slug'], axis = 1)
    return df


def convert_times(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the date related columns into timeframes in seconds and categories for month/weekday/day_hour"""

    launch_to_deadline = (df['deadline'] - df['launched_at'])
    creation_to_launch = (df['launched_at'] - df['created_at'])

    df['launch_to_deadline'] = launch_to_deadline
    df['creation_to_launch'] = creation_to_launch

    df['month_launch'] = pd.to_datetime(df['launched_at'], unit='s').dt.month_name()
    df['weekday_launch'] = pd.to_datetime(df['launched_at'], unit='s').dt.day_name()
    df['day_hour_launch'] = pd.to_datetime(df['launched_at'], unit='s').dt.hour

    df['month_deadline'] = pd.to_datetime(df['deadline'], unit='s').dt.month_name()
    df['weekday_deadline'] = pd.to_datetime(df['deadline'], unit='s').dt.day_name()
    df['day_hour_deadline'] = pd.to_datetime(df['deadline'], unit='s').dt.hour

    df = df.drop(['deadline', 'launched_at', 'created_at'], axis=1)
    return df


def convert_drop_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the target columns to binary 1/0 values and drop all rows that aren't successful or failed"""

    df = df.query('state == "successful" | state == "failed" ')
    df.loc[:, 'state'] = df.state.apply(lambda x: 0 if 'failed' in x else 1)

    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows, and rows that are duplicated and rows that only differ in usd_type"""

    df = df[~(df['id'].duplicated() & (df['usd_type'] == "domestic"))]
    df = df.drop_duplicates('id', keep='first')
    return df


def main():

    # read data
    raw_dir = os.path.join(os.path.dirname(__file__), '..', 'data/raw')
    processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data/processed')
    data = pd.read_csv(os.path.join(raw_dir, 'kickstarter.csv'))


    # convert columns 
    data = convert_goal_to_usd(data)
    data = convert_backers_pledged(data)
    data = convert_string_wordcount(data)
    data = convert_categories(data)
    data = convert_times(data)

    data = convert_drop_labels(data)
    data = drop_duplicates(data)

    drop_these = [
        'converted_pledged_amount',
        'currency',
        'currency_symbol',
        'currency_trailing_code',
        'current_currency',
        'disable_communication',
        'friends',
        'fx_rate',
        'id',
        'is_backing',
        'is_starrable',
        'is_starred',
        'permissions',
        'pledged',
        'slug',
        'source_url',
        'spotlight',
        'state_changed_at',
        'urls',
        ]

    drop_these_too = [
        'creator',
        'location',
        'photo',
        'profile',
        ]

    data.drop(drop_these, axis=1, inplace=True)
    data.drop(drop_these_too, axis=1, inplace=True)


    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    data.to_csv(os.path.join(processed_dir, 'kickstarter_clean.csv'), index=False)


if __name__ == "__main__":
    main()
