import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from constants import ZONES, DAYLIGHT_START, DAYLIGHT_END

def load_metered_data(year, save_clean=False):
    '''
    Process data into pivotted form
    
    PARAMETERS
    ----------
    year : int (default 2024)
        year to load data ffrom
    save_clean : bool (default False)
        save a cleaned version of rawdata, along with the processed one
    '''

    # Load input file
    file_path = f"data/data_metered_raw/hrl_load_metered_{year}.csv"
    df = pd.read_csv(file_path)
    df["datetime_beginning_utc"] = pd.to_datetime(df["datetime_beginning_utc"])
    df["datetime_beginning_ept"] = pd.to_datetime(df["datetime_beginning_ept"])

    # Remove duplicated hour
    mask_fall_dst = (df["datetime_beginning_utc"] == pd.Timestamp(f"{year}-11-{DAYLIGHT_END[year]} 05:00:00"))
    df_cleaned = df[~mask_fall_dst].copy()

    # Duplicate missing hour
    spring_gap_time = pd.Timestamp(f"{year}-03-{DAYLIGHT_START[year]} 01:00:00")
    spring_insert_time = pd.Timestamp(f"{year}-03-{DAYLIGHT_START[year]} 02:00:00")
    spring_rows = df_cleaned[df_cleaned["datetime_beginning_ept"] == spring_gap_time].copy()
    spring_rows["datetime_beginning_ept"] = spring_insert_time
    spring_rows["datetime_beginning_utc"] = spring_rows["datetime_beginning_utc"] + pd.Timedelta(hours=1)
    df_fixed = pd.concat([df_cleaned, spring_rows], ignore_index=True)
    df_fixed = df_fixed.sort_values(by=["datetime_beginning_ept", "load_area"]).reset_index(drop=True)

    # Save the fixed hourly dataset
    if (save_clean):
        output_path = f"data/data_metered/by_year/hrl_load_metered_{year}_fixed.csv"
        df_fixed.to_csv(output_path, index=False)
        print("Cleaned hourly file saved to:", output_path)

    # Extract date and hour for pivoting
    df_load = df_fixed[["datetime_beginning_ept", "load_area", "mw"]].copy()
    df_load["date"] = df_load["datetime_beginning_ept"].dt.date
    df_load["hour"] = df_load["datetime_beginning_ept"].dt.hour

    # Loop through each load_area
    count = 0
    for area, df_area in df_load.groupby("load_area"):

        # Pivot to wide format: one row per date, 24 columns for each hour + rename columns
        daily_pivot = df_area.pivot_table(index="date", columns="hour", values="mw").sort_index()
        daily_pivot.columns = [f"H{int(h):02d}" for h in daily_pivot.columns]
        daily_pivot = daily_pivot.reset_index()

        # Save to CSV
        out_dir = Path(f"data/data_metered/by_zone/{area}")
        out_dir.mkdir(exist_ok=True)
        daily_pivot.to_csv(f"{out_dir}/{area}_daily_load_{year}.csv", float_format="%.3f")
        count += 1

        print(f"Saved: {out_dir}/{area}_daily_load_{year}.csv")
    print(f"All csv generated for {count} successfully!")


if __name__ == '__main__':
    for year in range(2022, 2026):
        load_metered_data(year)
