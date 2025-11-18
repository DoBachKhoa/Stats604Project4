# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Testing pipeline with datas from prior to 2025
#
# This notebooks test different pipelines with data from prior to 2025 (mainly 2024). We run the pipeline against the task of predicting electric load of 2024, 10 days that ends with saturday after Black Friday (the same setting as that for this year).

# %%
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from src.pipeline import ZeroPipeline, BaseMeanPipeline, BaseMeanByDayPipeline, \
                         BasicRegressionPipeline, PCARegressionPipeline, PCAWeatherRegressionPipeline, Faraday, Edison, Ampere
from src.data_loader import format_request, get_electric_data, get_weather_data
from src.utils import slide_week_day
from src.constants import HOURS, PRED_WEEK_START, WEATHER_FEATURES, ZONES

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
# Defining some available pipelines to play with
pipelines = {
    'zero' : (ZeroPipeline, dict()),
    'basemean8' : (BaseMeanPipeline, {'use_week': 8}), # More bias
    'basemean3' : (BaseMeanPipeline, {'use_week': 3}), # More variance 
    'basemeanbyday8' : (BaseMeanByDayPipeline, {'use_week': 8}), # More bias
    'basemeanbyday3' : (BaseMeanByDayPipeline, {'use_week': 3}), # More variance 
    'basereg3year' : (BasicRegressionPipeline, {'train_year': 3}),
    'basereg1year' : (BasicRegressionPipeline, {'train_year': 1}),
    'pca3year' : (PCARegressionPipeline, {'train_year': 3, 'train_year_pca': 3, 'num_PC': 3}),
    'candidate1' : (PCAWeatherRegressionPipeline, {'train_year': 3, 'train_year_pca': 3, 'num_PC': 5, \
                                                   'pca_input_dir' : 'data/data_weather_hourly_processed'}),
    'Faraday' : (Faraday, {'train_year': 3, 'train_year_pca': 3, 'num_PC': 5, \
                           'pca_input_dir' : 'data/data_weather_hourly_processed'}),
    'Edison' : (Edison, {'train_year': 3, 'train_year_pca': 3, 'num_PC': 5, \
                         'pca_input_dir' : 'data/data_weather_hourly_processed', 'param_dir': 'pca_params/global_params_2024'}),
    'Ampere' : (Ampere, {'train_year': 3, 'train_year_pca': 3, 'num_PC': 5, \
                         'correction_days': [[3, 0], [4, 0], [5, 0]],\
                         'pca_input_dir' : 'data/data_weather_hourly_processed', 'param_dir': 'pca_params/global_params_2024'}),
                         # Correcting last 3 days in the 10 day periods
    'AmpereAll' : (Ampere, {'train_year': 3, 'train_year_pca': 3, 'num_PC': 5, \
                            'correction_days': slide_week_day(-1, 1),\
                            'pca_input_dir' : 'data/data_weather_hourly_processed', 'param_dir': 'pca_params/global_params_2024'})
                            # Correcting every day in the 10 day periods
}

# %% [markdown]
# The following cell tests different pipelines on predicting a zone. By looking at the plots and the numbers for different zones, we gain insights into the strengths and weaknesses of different models. 

# %%
# Testing result of different pipelines on prediction of a zone for the year 2024
zone = 'AP'
year = 2024
pipeline_names = ['basemeanbyday8','pca3year', 'candidate1','Faraday','Edison','AmpereAll']
tick_positions = range(12, 325, 24)
day_labels = ['sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat',
              'sun', 'mon', 'tue', 'wed', 'TGV', 'BFD', 'sat']
request = {year: slide_week_day(-1, 1)}
ground_truth = np.array(get_electric_data(zone, request, daystart=PRED_WEEK_START)[HOURS])

# Create one subplot row per pipeline
fig, axes = plt.subplots(len(pipeline_names), 1, figsize=(12, 4 * len(pipeline_names)))
if len(pipeline_names) == 1: axes = [axes]

# Loop through the pipelines
for ax, pipeline_name in zip(axes, pipeline_names):

    # Build & train pipeline
    pipeline_class, pipeline_param = pipelines[pipeline_name]
    pipeline = pipeline_class(zone=zone, year=year, **pipeline_param)
    pipeline.train_model()

    print(f'Testing pipeline {pipeline.__class__.__name__} ...', end=' ')
    output = pipeline.test_model()
    lost1, lost2, lost3, lost_ratios = output['lost1'], output['lost2'], output['lost3'], output['lost_ratios'] 
    lost_ratios = [np.round(r, 2) for r in lost_ratios]

    # Run prediction
    request = {year: slide_week_day(-1, 1)}
    prediction = pipeline.predict_request(request)

    # Ploting
    ax.plot(ground_truth.flatten(), label='Ground truth', linewidth=1.5, color='tab:orange', alpha=0.5)
    ax.plot(prediction.flatten(), label='Prediction', linewidth=1.5, color='tab:blue')

    # Vertical boundaries
    for boundary in range(0, 360, 24):
        if boundary not in [96, 168]: ax.axvline(boundary, linestyle=':', linewidth=0.8, color='gray')
        else: ax.axvline(boundary, linewidth=1.2, color='gray')

    # Tick positions, labels & title
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(day_labels)
    ax.tick_params(axis='x', length=0)
    ax.set_title(f'{pipeline_name}: zone {zone}, year {year}, loss: {np.round(lost1, 2), lost2, lost3}\nDistribution: {lost_ratios}')
    ax.legend()

plt.tight_layout()
plt.show()


# %%
# Plot weather data in 2024
weather_data = get_weather_data(zone, request, daystart=PRED_WEEK_START)
fig, axes = plt.subplots(4, 1, figsize=(12, 8))
tick_positions = list(range(14))

# Temperature
for temp_feature in ['tavg', 'tmax', 'tmin']:
    axes[0].plot(weather_data[temp_feature], label=temp_feature)
for boundary in [3.5, 6.5]: axes[0].axvline(boundary, linewidth=1.2, color='gray')
axes[0].set_xticks(tick_positions)
axes[0].set_xticklabels(day_labels)
axes[0].tick_params(axis='x', length=0)
axes[0].set_title('Daily Temperature')
axes[0].legend()

# Precipitation
axes[1].plot(weather_data['prcp'])
for boundary in [3.5, 6.5]: axes[1].axvline(boundary, linewidth=1.2, color='gray')
axes[1].set_xticks(tick_positions)
axes[1].set_xticklabels(day_labels)
axes[1].tick_params(axis='x', length=0)
axes[1].set_title('Daily precipitation')

# Snow
axes[2].plot(weather_data['snow'])
for boundary in [3.5, 6.5]: axes[2].axvline(boundary, linewidth=1.2, color='gray')
axes[2].set_xticks(tick_positions)
axes[2].set_xticklabels(day_labels)
axes[2].tick_params(axis='x', length=0)
axes[2].set_title('Daily snow')

# Wind
axes[3].plot(weather_data['wspd'])
for boundary in [3.5, 6.5]: axes[3].axvline(boundary, linewidth=1.2, color='gray')
axes[3].set_xticks(tick_positions)
axes[3].set_xticklabels(day_labels)
axes[3].tick_params(axis='x', length=0)
axes[3].set_title('Daily wind speed')

plt.tight_layout()
plt.show()

# %%
WEATHER_FEATURES

# %% [markdown]
# Below we look at all zones to see the fitting pattern of the algorithm. The distribution of MSE across days also gives insight into which days are harder for the model to learn. Under assumptions, this can mean "which days is the holiday effect most apparent". This is helpful when analysizing strategies to account for holiday effects.

# %%
# For different zone, which day is hardest to fit in 2024?
zones = ZONES
year = 2024
pipeline_name = 'AmpereAll' # 'AmpereAll' for holiday-accounted model; 'candidate1' for holiday-unaccounted model.
tick_positions = range(12, 325, 24)
day_labels = ['sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat',
              'sun', 'mon', 'tue', 'wed', 'TGV', 'BFD', 'sat']
request = {year: slide_week_day(-1, 1)}
total_ratios = np.array([0]*10, dtype='float')
max_day = [0]*10

# Create one subplot row per pipeline
fig, axes = plt.subplots(len(zones), 1, figsize=(12, 4 * len(zones)))
if len(zones) == 1: axes = [axes]

# Loop through the pipelines
for ax, zone in zip(axes, zones):
    ground_truth = np.array(get_electric_data(zone, request, daystart=PRED_WEEK_START)[HOURS])

    # Build & train pipeline
    pipeline_class, pipeline_param = pipelines[pipeline_name]
    pipeline = pipeline_class(zone=zone, year=year, **pipeline_param)
    pipeline.train_model()

    # Test pipeline
    output = pipeline.test_model()
    lost1, lost2, lost3, lost_ratios = output['lost1'], output['lost2'], output['lost3'], output['lost_ratios'] 
    lost_ratios = [np.round(r, 2) for r in lost_ratios]
    total_ratios = total_ratios + np.array(lost_ratios)
    max_day[output['1stHigh']] += 1
    max_day[output['2ndHigh']] += 1

    # Run prediction
    request = {year: slide_week_day(-1, 1)}
    prediction = pipeline.predict_request(request)

    # Ploting
    ax.plot(ground_truth.flatten(), label='Ground truth', linewidth=1.5, color='tab:orange', alpha=0.5)
    ax.plot(prediction.flatten(), label='Prediction', linewidth=1.5, color='tab:blue')

    # Vertical boundaries
    for boundary in range(0, 360, 24):
        if boundary not in [96, 168]: ax.axvline(boundary, linestyle=':', linewidth=0.8, color='gray')
        else: ax.axvline(boundary, linewidth=1.2, color='gray')

    # Tick positions, labels & title
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(day_labels)
    ax.tick_params(axis='x', length=0)
    ax.set_title(f'{pipeline_name}: zone {zone}, year {year}, loss: {np.round(lost1, 2), lost2, lost3}\nDistribution: {lost_ratios}')
    ax.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# By plotting the lost ratio and the peak days, we can see the peak day distribution across zones, as well as which days cause troble for the model to learn.

# %%
# Plot accumulated lost ratios for all regions
x_positions = list(range(10))
plt.bar(x_positions, list(total_ratios))
plt.xticks(x_positions, day_labels[4:])
plt.show()

# Plot accumulated peak days for all regions
x_positions = list(range(10))
plt.bar(x_positions, list(max_day))
plt.xticks(x_positions, day_labels[4:])
plt.show()

# %% [markdown]
# We can also plot the peak days of the same period in pass years, and run a model, to see which days peak and/or are hard to learn. Due to missing weather data, hear we do this for the mean-by-weakday model.

# %%
# How about peak days the years before 2025?
# Weather data are not complete from 2021 backwards, so we revert to mean models that does not need it
zones = ZONES
years = range(2019, 2025)
pipeline_name = 'basemeanbyday8' # 'basemeanbyday8' 'basereg1year'
tick_positions = range(12, 325, 24)
day_labels = ['sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat',
              'sun', 'mon', 'tue', 'wed', 'TGV', 'BFD', 'sat']
request = {year: slide_week_day(-1, 1)}

for year in years:
    total_ratios = np.array([0]*10, dtype='float')
    max_day = [0]*10
    
    # Loop through the pipelines
    for zone in zones:
        ground_truth = np.array(get_electric_data(zone, request, daystart=PRED_WEEK_START)[HOURS])
    
        # Build & train pipeline
        pipeline_class, pipeline_param = pipelines[pipeline_name]
        pipeline = pipeline_class(zone=zone, year=year, **pipeline_param)
        pipeline.train_model()
    
        # Test pipeline
        output = pipeline.test_model()
        lost1, lost2, lost3, lost_ratios = output['lost1'], output['lost2'], output['lost3'], output['lost_ratios'] 
        lost_ratios = [np.round(r, 2) for r in lost_ratios]
        total_ratios = total_ratios + np.array(lost_ratios)
        max_day[output['1stHigh']] += 1
        max_day[output['2ndHigh']] += 1
    
        # Run prediction
        request = {year: slide_week_day(-1, 1)}
        prediction = pipeline.predict_request(request)
    
    x_positions = list(range(10))
    _, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(x_positions, list(total_ratios))
    axes[0].set_xticks(x_positions, day_labels[4:])
    axes[1].bar(x_positions, list(max_day))
    axes[1].set_xticks(x_positions, day_labels[4:])
    plt.suptitle(f'Lost distribution and peak day distribution in year {year}')
    plt.tight_layout()
    plt.show()

# %%
