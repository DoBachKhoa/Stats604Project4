import os
import numpy as np
import pandas as pd
import requests
import pickle
from tqdm import tqdm
from sklearn.decomposition import PCA
from datetime import date as Date
from datetime import datetime
from meteostat import Point, Daily, Hourly
from meteostat import Stations
from src.fetch_weather import fillna_by_month
from src.utils import add_relative_week_column
from src.constants import WEATHER_FEATURES, ZONES, MONTH, LEAPS, PRED_WEEK_START, \
                          WEATHER_FEATURES_HOURLY, WEATHER_FEATURES_HOURLY_COUNTS, HOURS

def hourly_open_meteo(zone, counts, date=None):
    """
    Hourly forecast for a single calendar date at (lat, lon) in LOCAL time.
    Returns a list of dicts, one per hour.
    """
    # Look up coordinate of zone
    df_coords = pd.read_csv('data/zone_locations.csv')[['zone', 'lon', 'lat']]
    lon_coord = df_coords.loc[df_coords['zone']==zone]['lon'].iloc[0]
    lat_coord = df_coords.loc[df_coords['zone']==zone]['lat'].iloc[0]

    # Search station
    stations = Stations().nearby(lat_coord, lon_coord)
    station = stations.fetch(1)
    lat_coord_station, long_coord_station = station[["latitude","longitude"]].iloc[0]

    # Fetch weather predictions
    if date is None: date = Date.today()
    hourly_vars = [
        "temperature_2m", "dewpoint_2m", "relative_humidity_2m",
        "precipitation", "windspeed_10m", "pressure_msl"]
    params = {
        "latitude": lat_coord_station,
        "longitude": long_coord_station,
        "hourly": ",".join(hourly_vars),
        "timezone": "auto",                     # local tz for that point
        "start_date": date.isoformat(),
        "end_date": date.isoformat(),            # only that day
    }
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    hz = data.get("hourly", {})
    times = hz.get("time", [])
    if not times:
        raise ValueError("No hourly data returned for that date/location.")

    # Build per-hour records
    fetch_data = pd.DataFrame(hz)
    output = dict()
    for feature, code in zip(hourly_vars, WEATHER_FEATURES_HOURLY):
        pca_dir = f'data/data_weather_hourly_processed/pcas/{zone}_{code}.pkl'
        with open(pca_dir, 'rb') as file:
            pca = pickle.load(file)
        feature_data = np.array(fetch_data[feature]).reshape(1, -1)
        feature_data = pca.transform(feature_data)[:, :counts[code]]
        for i in range(counts[code]):
            output[f'{code}_PC{i}'] = float(feature_data[0, i])

    output = pd.DataFrame([output])
    output['date'] = date
    output['year'] = date.year
    output = add_relative_week_column(output, 'date', date.year, week_start=PRED_WEEK_START)
    return output

# Example:
hours = hourly_open_meteo('AECO', WEATHER_FEATURES_HOURLY_COUNTS)
print(hours)
