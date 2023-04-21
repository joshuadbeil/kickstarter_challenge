import os
import pandas as pd
from tqdm import tqdm

def main():
    """run this file in the terminal to generate a new merged 'kickstarter.csv' from combining the 56 smaller csv files.
    """

    # set path ../data/raw relative to the import.py location
    raw_dir = os.path.join(os.path.dirname(__file__), '..', 'data/raw')

    # find all KickstarterXXX.csv files in that location
    csv_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.csv') and f.startswith('Kickstarter') and len(f)==18]


    df_list = []
    # wrap it in a progress bar to make it look a little more fancy
    with tqdm(total=len(csv_files), desc="Processing CSV files") as pbar:
        for filename in csv_files:
            df_import = pd.read_csv(filename)
            df_list.append(df_import)
            pbar.update(1)

    data = pd.concat(df_list, ignore_index=True)

    # write 'kickstarter.csv' into '/data/raw' into the raw folder
    with tqdm(total=1, desc="Exporting to CSV") as pbar:
        data.to_csv(os.path.join(raw_dir, 'kickstarter.csv'), index=False)
        pbar.update(1)

    print("Done! kickstarter.csv written into /data/raw/kickstarter.csv")

if __name__ == "__main__":
    main()
