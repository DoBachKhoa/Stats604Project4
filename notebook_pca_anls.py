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
# # Encoding metered data with pca
#
# We look at encoding metered data across region using pca. If possible, this would allows for borrowing information across zones to account for hard-to-learn effects, such as holiday effects.

# %%
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from src.pipeline import ZeroPipeline, BaseMeanPipeline, BaseMeanByDayPipeline, \
                         BasicRegressionPipeline
from src.data_loader import format_request, get_electric_data, get_weather_data
from src.utils import slide_week_day
from src.constants import HOURS, PRED_WEEK_START, WEATHER_FEATURES, ZONES

# %% [markdown]
# The cell below collects all metered data of 2022-2024 across regions in a 13 week period, from week -10 to week 2 (relative to week 0 which is the thanksgiving week). We admit that there are potential 'offsides' for using thankgiving data in 2024 to learn this encoding, but we argue that the bias from a PCA encoder is negligible.

# %%
request = {year: slide_week_day(-10, 3, daystart=PRED_WEEK_START) for year in [2022, 2023, 2024]}
electric_data = None
variances = dict()
for zone in ZONES:
    temp = get_electric_data(zone, request, daystart=PRED_WEEK_START)
    hours_np = np.array(temp[HOURS])
    variance = np.sum(hours_np**2)
    variances[zone] = variance
    temp[HOURS] = hours_np/np.sqrt(variance)
    if electric_data is None: electric_data = temp
    else: electric_data = pd.concat([electric_data, temp], ignore_index=True)

# %%
electric_data

# %%
hour_data = np.array(electric_data[HOURS])

# %%
pca = PCA(n_components=24)
pca.fit(hour_data)
pca.explained_variance_ratio_

# %%
np.cumsum(pca.explained_variance_ratio_*100)

# %%
plt.plot(pca.explained_variance_ratio_)
plt.show()

# %% [markdown]
# The following cell plots the most dominant PC's.

# %%
plt.plot(pca.components_[0], label = 'PC 0')
plt.plot(pca.components_[1], label = 'PC 1')
plt.plot(pca.components_[2], label = 'PC 2')
plt.legend()
plt.show()

# %%
# Projections
for zone in ZONES:
    num_coms = [5]
    new_request = {2024: slide_week_day(-1, 1)}
    new_electric_data = get_electric_data(zone, new_request, daystart=PRED_WEEK_START)
    new_electric_data_np = np.array(new_electric_data[HOURS])
    projections = [new_electric_data_np.copy()]
    for l in num_coms:
        new_electric_data_proj = pca.transform(new_electric_data_np/np.sqrt(variances[zone]))[:, :l]
        new_electric_data_proj = (new_electric_data_proj @ pca.components_[:l] + pca.mean_) * np.sqrt(variances[zone])
        projections.append(new_electric_data_proj)
    
    # Reconstruct data
    names = ['Original']+[f'W{i}PC' for i in num_coms]
    plt.figure(figsize=(15, 5))
    for proj, name in zip(projections, names):
        plt.plot(proj.flatten(), label=name)
    
    # Plot mean
    mean_days = np.repeat(pca.mean_[None, :], 14, axis=0).ravel()
    plt.plot(mean_days, color='gray', linewidth=1., alpha=0.9, label='mean')
    
    # Vertical boundaries
    for boundary in range(0, 360, 24):
        if boundary not in [96, 168]: plt.axvline(boundary, linestyle=':', linewidth=0.8, color='gray')
        else: plt.axvline(boundary, linewidth=1.2, color='gray')
    
    # Tick positions, labels & title
    tick_positions = range(12, 325, 24)
    day_labels = ['sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat',
                  'sun', 'mon', 'tue', 'wed', 'TGV', 'BFD', 'sat']
    plt.xticks(tick_positions, day_labels)
    plt.tick_params(axis='x', length=0)

    plt.title(zone)
    plt.legend()
    plt.show()

# %%
zone = 'AP'
request_tgv = {year: slide_week_day(0, 0, 3, 4, daystart=PRED_WEEK_START) for year in range(2022, 2025)}
electric_data_tgv = get_electric_data(zone, request_tgv, daystart=PRED_WEEK_START)
electric_data_tgv_np = np.array(electric_data_tgv[HOURS])
print(f"Mean magnitute: {np.round(np.mean(electric_data_tgv_np), 3)}")

predict_0 = np.array([pca.mean_, pca.mean_, pca.mean_])*np.sqrt(variances[zone])
lost_0 = np.sum((predict_0-electric_data_tgv_np)**2)/240
print(f"Loss 0: {np.round(np.sqrt(lost_0), 3)}")
for i in range(24):
    predict_i = pca.transform(electric_data_tgv_np/np.sqrt(variances[zone]))[:, :i+1]
    predict_i = (predict_i @ pca.components_[:i+1] + pca.mean_)*np.sqrt(variances[zone])
    lost_i = np.sum((predict_i-electric_data_tgv_np)**2)/240
    print(np.round(100*(1-lost_i/lost_0), 3), np.round(np.sqrt(lost_i), 3))

# %% [markdown]
# **We can see that althought different zone has different magnitudes, a global PCA is possible and the most dominant PCs explains all the region reasonably well. This prompts the use of using a global PCA to allow for shared information and learning between the regions.**

# %%
