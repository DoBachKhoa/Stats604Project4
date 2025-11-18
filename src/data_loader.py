'''
Define the helper data loader functions 
for the pipelines' convenicence of loading the needed data
E.g.1: 
    request = {2024: [[3, 0], [4, 0], [5, 0]]} # Weeks 0 means thanksgiving week,
                                               # Days 3, 4, 5 means thu, fri, sat
    electric_data = get_electric_data(request) # Returns pd.df of electric data in requested dates

E.g.2: 
    request = {2023: utils.slide_week_day(-1, 1)} # Request all days in the week -1 and 0
                                                  # which are the week before and thanksgiving week
    weather_data = get_weather_data(request)      # Returns pd.df of weather data in requested dates
'''
import numpy as np
import pandas as pd
from src.constants import PRED_WEEK_START

def format_request(request):
    request_days = pd.DataFrame(columns=['year', 'relative_week', 'day_of_week'])
    for year, day_and_week in request.items():
        days = [d for (d, _) in day_and_week]
        weeks = [w for (_, w) in day_and_week]
        days_year = pd.DataFrame({'day_of_week': days, 'relative_week': weeks})
        days_year['year'] = year
        request_days = pd.concat([request_days, days_year], ignore_index=True)
    return request_days

def get_electric_data(zone, request, daystart=PRED_WEEK_START):
    if daystart != PRED_WEEK_START: raise NotImplementedError
    request_days = format_request(request)
    data = pd.read_csv(f'data/data_metered_processed/metered_data_{zone}.csv', index_col=0)
    return data.merge(request_days, on=['year', 'relative_week', 'day_of_week'], how='inner')

def get_weather_data(zone, request, daystart=PRED_WEEK_START):
    if daystart != PRED_WEEK_START: raise NotImplementedError
    request_days = format_request(request)
    data = pd.read_csv(f'data/data_weather_daily/weather_data_{zone}.csv', index_col=0)
    return data.merge(request_days, on=['year', 'relative_week', 'day_of_week'], how='inner')

def get_weather_data_hourly(zone, request, daystart=PRED_WEEK_START, \
                            data_dir = 'data/data_weather_hourly_processed'):
    if daystart != PRED_WEEK_START: raise NotImplementedError
    request_days = format_request(request)
    data = pd.read_csv(f'{data_dir}/weather_data_pca_{zone}.csv', index_col=0)
    return data.merge(request_days, on=['year', 'relative_week', 'day_of_week'], how='inner')