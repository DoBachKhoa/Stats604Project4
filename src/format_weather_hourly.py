import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from src.constants import ZONES, HOURS, WEATHER_FEATURES_HOURLY

def format_weather_hourly_pca(features, counts, zones=ZONES):

    # Format counts
    if isinstance(counts, int):
        counts = [counts]*len(features)
    if isinstance(counts, dict):
        counts = [counts[feature] for feature in features]

    # Loop through all zones
    variance_explain ={feature: 0. for feature in features}
    for zone in zones:
        data = None
        adding_columns = dict()
        for _, (feature, count) in enumerate(zip(features, counts)):
            directory = f'data/data_weather_hourly/{feature}/weather_{zone}.csv'
            data_temp = pd.read_csv(directory)
            if data is None: data = data_temp[['year', 'relative_week', 'day_of_week']].copy()
            data_temp = np.array(data_temp[HOURS])
            pca = PCA(n_components=24)
            pca.fit(data_temp)
            data_transformed = pca.transform(data_temp)[:, :count]
            pca_dir = 'data/data_weather_hourly_processed/pcas'
            os.makedirs(pca_dir, exist_ok=True)
            with open(f'{pca_dir}/{zone}_{feature}.pkl', mode='wb') as f:
                pickle.dump(pca, f)
            for j in range(count):
                adding_columns[f'{feature}_PC{j}'] = data_transformed[:, j]
            if np.sum(pca.explained_variance_) < 1e-6:
                variance_explain[feature] += 10
            else: variance_explain[feature] += np.sum(pca.explained_variance_ratio_[:count])/len(zones)
        data = pd.concat([data, pd.DataFrame(adding_columns, index=data.index)], axis=1)
        data.to_csv(f'data/data_weather_hourly_processed/weather_data_pca_{zone}.csv')

    return variance_explain

if __name__ == '__main__':
    counts = {
        'temp': 2,
        'dwpt': 2,
        'rhum': 1,
        'prcp': 4,
        'wspd': 1,
        'pres': 1
    }
    json_name = 'data/data_weather_hourly_processed/counts_json'
    with open(json_name, 'w') as f:
        json.dump(counts, f, indent=4)
    variance_explain = format_weather_hourly_pca(features=list(counts.keys()), counts=counts, zones=ZONES)
    print(variance_explain)

