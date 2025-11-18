'''
File to load and store hourly feature data.
For each zone and each feature (such as temerature),
data are store as a csv file
where the rows are the dates (across years - we take from 2022 to now)
and the columns are the 24 hours, the relative week (compared to thanks giving week),
and the relative day of week (0 for monday to 6 for saturday).
We set a week to begin with saturday (meaning 6, 0, 1, 2, 3, 4, 5), due to
saturday being the last prediction day.
'''
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from meteostat import Point, Hourly
from meteostat import Stations
from src.fetch_weather import fillna_by_month
from src.utils import add_relative_week_column
from src.constants import ZONES, PRED_WEEK_START, WEATHER_FEATURES_HOURLY, HOURS

def fetch_weather_hourly_feature(year, zone, feature='temp', fillna=False, df_coords=None, week_start=PRED_WEEK_START):
    # Load zone coordinate data
    if df_coords is None: df_coords = pd.read_csv('data/zone_locations.csv')[['zone', 'lon', 'lat']]
    start = datetime(year, 1, 1)
    end = datetime(year+1, 1, 1) if year < 2025 else datetime(year, 10, 31)

    # Look up coordinate of zone
    lon_coord = df_coords.loc[df_coords['zone']==zone]['lon'].iloc[0]
    lat_coord = df_coords.loc[df_coords['zone']==zone]['lat'].iloc[0]

    # Search station
    stations = Stations().nearby(lat_coord, lon_coord)
    station = stations.fetch(1)
    lat_coord_station, long_coord_station = station[["latitude","longitude"]].iloc[0]
    location = Point(lat_coord_station ,long_coord_station)

    # Get daily data for that year
    data = Hourly(location, start, end)
    data = data.fetch()
    data.reset_index(inplace=True, names='time')
    data = data[['time', feature]].copy()
    data = data.iloc[:-1]

    # Pivot data
    data["date"] = data["time"].dt.date
    data["hour"] = data["time"].dt.hour
    daily_pivot = data.pivot_table(index="date", columns="hour", values=feature, dropna=False)\
                  .sort_index().astype("float64")
    daily_pivot.columns = [f"H{int(h):02d}" for h in daily_pivot.columns]
    daily_pivot = daily_pivot.reset_index()
    daily_pivot['year'] = year
    if fillna:
        if feature in ['temp', 'dwpt', 'rhum', 'wspd', 'pres', 'tsun']:
            for hour in HOURS:
                daily_pivot = fillna_by_month(daily_pivot, hour, 'date')
        elif feature in ['prcp', 'snow']:
            daily_pivot.fillna(0, inplace=True)
        else:
            assert False, "Invalid weather feature"
    daily_pivot = add_relative_week_column(daily_pivot, 'date', year, week_start=week_start)
    daily_pivot.drop('date', axis=1, inplace=True)
    if daily_pivot.isna().any().any() and fillna:
        print(year, zone, feature)
        assert False, "There are still na's in the data!"
    return daily_pivot

def load_weather_hourly(years=[2022, 2023, 2024, 2025], zones=ZONES, features=WEATHER_FEATURES_HOURLY, \
                        fillna=False, df_coords=None, week_start=PRED_WEEK_START):
    for feature in tqdm(features):
        directory = f'data/data_weather_hourly_raw/{feature}'
        os.makedirs(directory, exist_ok=True)
        for zone in zones:
            data_all = None
            for year in years:
                data_year = fetch_weather_hourly_feature(year, zone, feature, fillna=fillna, \
                                                          df_coords=df_coords, week_start=week_start)
                if data_year.isna().any().any() and fillna:
                    print(year, zone, feature)
                    assert False, "There are still na's in the data!"
                if data_all is None: data_all = data_year.copy()
                else: data_all = pd.concat([data_all, data_year], ignore_index=True)
            if data_all.isna().any().any() and fillna:
                print(year, zone, feature)
                assert False, "There are still na's in the data!"
            data_all.to_csv(directory+f'/weather_{zone}.csv')


if __name__ == '__main__':
    load_weather_hourly(features=WEATHER_FEATURES_HOURLY, fillna=True)

