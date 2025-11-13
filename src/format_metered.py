import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from utils import add_relative_week_column
from constants import ZONES, DAYLIGHT_START, DAYLIGHT_END, PRED_WEEK_START

def load_metered_data(year=2024, save_file=False, week_start=PRED_WEEK_START):
    '''
    Process data into pivotted form
    
    PARAMETERS
    ----------
    year : int (default 2024)
        year to load data from
    save_file : bool (default False)
        save the loaded data as csv file, in addition to just returning
    week_start : int (default PRED_WEEK_START)
        indicate the day that starts the week.
        0 means Monday, 6 means Sunday.
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

    # # Save the fixed hourly dataset
    # output_path = f"data/data_metered/by_year/hrl_load_metered_{year}_fixed.csv"
    # df_fixed.to_csv(output_path, index=False)
    # print("Cleaned hourly file saved to:", output_path)

    # Extract date and hour for pivoting
    df_load = df_fixed[["datetime_beginning_ept", "load_area", "mw"]].copy()
    df_load["date"] = df_load["datetime_beginning_ept"].dt.date
    df_load["hour"] = df_load["datetime_beginning_ept"].dt.hour

    # Loop through each load_area
    count = 0
    output = dict()
    for area, df_area in df_load.groupby("load_area"):

        # Pivot to wide format: one row per date, 24 columns for each hour + rename columns
        daily_pivot = df_area.pivot_table(index="date", columns="hour", values="mw").sort_index()
        daily_pivot.columns = [f"H{int(h):02d}" for h in daily_pivot.columns]
        daily_pivot = daily_pivot.reset_index()
        daily_pivot = add_relative_week_column(daily_pivot, 'date', year, week_start=week_start)
        daily_pivot['year'] = year
        daily_pivot.drop('date', axis=1, inplace=True)

        # Save to CSV
        if save_file:
            out_dir = "data/data_metered_processed"
            daily_pivot.to_csv(f"{out_dir}/metered_data_{area}_{year}.csv", float_format="%.3f")
            print(f"Saved: {out_dir}/metered_data_{area}_{year}.csv")
        output[area] = daily_pivot
        count += 1

    if save_file:
        print(f"Generated {count} csv file(s) successfully!")
    return output

def load_metered_data_multiple_years(years, week_start=PRED_WEEK_START):
    '''
    Process data into pivotted form
    
    PARAMETERS
    ----------
    year : list(int)
        list of years to load data from
    week_start : int (default PRED_WEEK_START)
        indicate the day that starts the week.
        0 means Monday, 6 means Sunday.
    '''
    # Prepare output
    col_names = [f"H{int(h):02d}" for h in range(24)]+['year', 'relative_week', 'day_of_week']
    output = {zone: pd.DataFrame(columns=col_names) for zone in ZONES}

    # Run load_metered_data function for all years
    for year in years:
        output_year = load_metered_data(year, save_file=False, week_start=week_start)
        for zone in ZONES:
            output[zone] = pd.concat([output[zone], output_year[zone]], ignore_index=True)

    # Save data in csv
    out_dir = "data/data_metered_processed"
    for zone in ZONES:
        output[zone].to_csv(f"{out_dir}/metered_data_{zone}.csv", float_format="%.3f")
        print(f"Saved: {out_dir}/metered_data_{zone}.csv")


if __name__ == '__main__':
    load_metered_data_multiple_years(list(range(2022, 2026)))
