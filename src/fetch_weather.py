'''
Fetch and format daily weather data
For each zone, data are store as a csv file
where the rows are the dates (across years - we take from 2022 to now)
and the columns are the weather features, the relative week (compared to thanks giving week),
and the relative day of week (0 for monday to 6 for saturday).
We set a week to begin with saturday (meaning 6, 0, 1, 2, 3, 4, 5), due to
saturday being the last prediction day.
'''
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from meteostat import Point, Daily
from meteostat import Stations
from src.utils import add_relative_week_column
from src.constants import WEATHER_FEATURES, ZONES, MONTH, LEAPS

def fillna_by_month(df: pd.DataFrame, value_col: str = "value", date_col: str = "datetime"):
    """
    Fill NA values in value_col with the mean of the corresponding month.

    Parameters
    ----------
    df : DataFrame with columns [date_col, value_col]
    value_col : str
        name of the numeric column containing temperatures
    date_col : str
        name of the datetime column

    Return
    ------
    pandas.DataFrame
        data frame after filled
    """
    # Copy data and ensure datetime dtype
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Grouping and filling
    key = df[date_col].dt.to_period("M")
    month_means = df.groupby(key, dropna=False)[value_col].transform("mean")
    filled = df[value_col].where(df[value_col].notna(), month_means)

    # If month is all NA
    overall_mean = df[value_col].mean()
    filled = filled.fillna(overall_mean)

    # Return
    df[value_col] = filled
    return df

def fetch_weather_data(year=2024, start_month=9, end_month=11, weather_features=WEATHER_FEATURES, zones=ZONES, fillna=False, df_coords=None):
    '''
    Fetch weather data of recent years from the meteostat library
    
    Parameters
    ----------
    year : int (default 2024)
        year to load data from. Recommend: 2021 to 2025.
        If the year is 2025, then only load till at most the end of October.
    start_month : int (default 9)
        starting month to load data from. 
        If the year is 2025, then could at most be 10.
    end_month : int (default 11)
        ending month to load data from.
        If the year is 2025, then only load till at most the end of October
    weather_features : list (str) (default WEATHER_FEATURES)
        list of weather features to be fetched.
    zones : list (str) (default ZONES)
        list of zones to be fetched.
    fillna: bool
        whether to fill nans with the monthly mean values (for tems, win spd, pres)
        or zeros (for snow and prcp)
    df_coords: optional pandas.DataFrame (default None)
        if None: load and use zone coordinate data.
        Give the option of passing zone coordinate data if preloaded.

    Return
    ------
    dictionary (str: pandas.DataFrame)
        fetched data from meteostat
    '''
    # Load zone coordinate data
    if df_coords is None: df_coords = pd.read_csv('data/zone_locations.csv')[['zone', 'lon', 'lat']]
    start = datetime(year, start_month, 1)
    if end_month == 2 and year in LEAPS: end_date = 29
    else: end_date = MONTH[end_month-1]
    end = datetime(year, end_month, end_date)

    # Load data for zones:
    output = dict()
    for zone in tqdm(zones):

        # Look up coordinate of zone
        lon_coord = df_coords.loc[df_coords['zone']==zone]['lon'].iloc[0]
        lat_coord = df_coords.loc[df_coords['zone']==zone]['lat'].iloc[0]

        # Search station
        stations = Stations().nearby(lat_coord, lon_coord)
        station = stations.fetch(1)
        lat_coord_station, long_coord_station = station[["latitude","longitude"]].iloc[0]
        location = Point(lat_coord_station ,long_coord_station)

        # Get daily data for that year
        data = Daily(location, start, end)
        data = data.fetch()
        data = data[weather_features]
        data.reset_index(inplace=True, names='date')

        # Fill na
        if fillna:
            for feature in weather_features:
                if feature in ['tmin', 'tavg', 'tmax', 'wspd', 'pres']:
                    data = fillna_by_month(data, feature, 'date')
                elif feature in ['prcp', 'snow']:
                    data[feature] = data[feature].fillna(0)
                else:
                    raise ValueError('Invalid weather feature')

        # Add daily data to output
        output[zone] = data

    # Return
    return output

def save_weather_data(weather_features=WEATHER_FEATURES, zones=ZONES, fillna=False, df_coords=None):
    '''
    Fetch and save weather data of recent years from the meteostat library
    Time: months 8 to 11 of the years 2022 to 2024, and months 8 to 10 of 2025

    Parameters
    ----------
    weather_features : list (str) (default WEATHER_FEATURES)
        list of weather features to be fetched.
    zones : list (str) (default ZONES)
        list of zones to be fetched.
    fillna: bool
        whether to fill nans with the monthly mean values (for tems, win spd, pres)
        or zeros (for snow and prcp)
    df_coords: optional pandas.DataFrame (default None)
        if None: load and use zone coordinate data.
        Give the option of passing zone coordinate data if preloaded.
    '''
    # Load zone coordinate data
    if df_coords is None: df_coords = pd.read_csv('data/zone_locations.csv')[['zone', 'lon', 'lat']]
    output = {zone: None for zone in zones}
    starts_ends = [[i, 1, 12] for i in range(2022, 2025)] + [[2025, 1, 10]]

    for year, start_month, end_month in starts_ends:
        output_year = fetch_weather_data(year=year, start_month=start_month, end_month=end_month, \
                                         weather_features=weather_features, zones=zones, fillna=fillna, df_coords=df_coords)
        for zone in zones:
            data_year = output_year[zone]
            data_year = add_relative_week_column(data_year, 'date', year, week_start=6)
            data_year['year'] = data_year['date'].dt.year
            data_year.drop('date', axis=1, inplace=True)
            if output[zone] is None: output[zone] = data_year.copy()
            else: output[zone] = pd.concat([output[zone], data_year], ignore_index=True)

    for zone in zones:
        output[zone].to_csv(f'data/data_weather_daily/weather_data_{zone}.csv')


if __name__ == '__main__':
    df_coords = pd.read_csv('data/zone_locations.csv')[['zone', 'lon', 'lat']]
    save_weather_data(weather_features=WEATHER_FEATURES, zones=ZONES, fillna=True, df_coords=df_coords)

