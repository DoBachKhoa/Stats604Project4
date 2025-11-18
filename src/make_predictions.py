import numpy as np
from datetime import date as Date, datetime, timedelta
from src.fetch_weather_predictions import hourly_open_meteo
from src.pipeline import BaseMeanByDayPipeline, Ampere
from src.utils import slide_week_day, select_argmax_window3
from src.constants import PRED_WEEK_START, ZONES, WEATHER_FEATURES_HOURLY_COUNTS

import urllib3
urllib3.disable_warnings()

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    tomorrow = Date.today() + timedelta(days=1)
    output_hour_load = []
    output_peak_day = []
    output_peak_all = []
    print(f'"{tomorrow}"', end='')
    for zone in ZONES:
        pipeline = Ampere(param_dir='pca_params/global_params_2024', correction_days=slide_week_day(-1, 1, daystart=PRED_WEEK_START), \
                        zone=zone, year=datetime.now().year, train_year=4, num_PC=5, pca_input_dir='data/data_weather_hourly_processed')
        pipeline.train_model()
        weather = hourly_open_meteo(zone, WEATHER_FEATURES_HOURLY_COUNTS, tomorrow)
        d, w = weather.iloc[0]['day_of_week'], weather.iloc[0]['relative_week']
        output = pipeline.predict(week=w, day=d, year=weather.iloc[0]['year'], weather=weather, with_correction=True)
        for load in output[0]: output_hour_load.append(load)
        output_peak_day.append(select_argmax_window3(output))
        output_peak_all.append(0 if [d, w] in [[5, -1], [6, 0] ,[3, 0], [4, 0]] else 1)
    for num in output_hour_load: print(f', {np.round(num, 4)}', end='')
    for num in output_peak_day: print(f', {num[0]}', end='')
    for num in output_peak_all: print(f', {num}', end='')
    print()

