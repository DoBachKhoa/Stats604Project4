import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from src.pipeline import PCAWeatherRegressionPipeline
from src.data_loader import get_electric_data, get_weather_data_hourly, format_request
from src.utils import slide_week_day, calculate_loss, to_string
from src.constants import HOURS, PRED_WEEK_START, WEATHER_FEATURES, ZONES, WEEKDAYS

def generate_pca_metered(request, param_path):

    # Get electric data from all the zones
    electric_data = None
    variances = dict()
    for zone in ZONES:
        temp = get_electric_data(zone, request, daystart=PRED_WEEK_START)
        hours_np = np.array(temp[HOURS])
        variance = np.sum(hours_np**2)
        variances[zone] = float(variance)
        temp[HOURS] = hours_np/np.sqrt(variance)
        if electric_data is None: electric_data = temp
        else: electric_data = pd.concat([electric_data, temp], ignore_index=True)

    # Calculate pca
    hour_data = np.array(electric_data[HOURS])
    pca = PCA(n_components=24)
    pca.fit(hour_data)

    # Save pca and variances
    with open(f'{param_path}/pca_global.pkl', 'wb') as file:
        pickle.dump(pca, file)
    with open(f'{param_path}/variances.json', 'w') as file:
        json.dump(variances, file)

class PCAGlobalPipeline(PCAWeatherRegressionPipeline):
    def __init__(self, zone='AECO', year=2024, train_year=3, train_year_pca=None, num_PC=3, \
                 pca_input_dir = 'data/data_weather_hourly_processed'):
        super().__init__(zone, year, train_year, train_year_pca, num_PC, pca_input_dir)
        self.metered_variance = 1.
    
    def _request_train_electric_data(self):
        output = {i: slide_week_day(-10, -2, daystart=PRED_WEEK_START) \
                  for i in range(self.year, self.year-self.train_year_pca, -1)}
        return output
    
    def train_model(self):
        # Prepare pca & load data
        if not self.trained_pca: assert False, 'pca has to be preload'
        electric_data = get_electric_data(self.zone, \
                                          self._request_train_electric_data(), \
                                          daystart=PRED_WEEK_START)
        weather_data = get_weather_data_hourly(self.zone, \
                                               self._request_train_weather_data(), \
                                               daystart=PRED_WEEK_START, data_dir=self.pca_input_dir)

        # Prepare x
        x_train = weather_data[self.input_features].copy()
        for i, weekday in enumerate(WEEKDAYS[:-1]):
            x_train[weekday] = (weather_data['day_of_week'] == i).astype(int)
        self.train_year_list = list(weather_data['year'].unique())[:-1]
        for year in self.train_year_list:
            x_train[f'year{year}'] = (weather_data['year'] == year).astype(int)

        # Prepare y
        electric_data_np = np.array(electric_data[HOURS])
        electric_data_proj = self.pca.transform(electric_data_np/np.sqrt(self.metered_variance))[:, :self.num_PC]
        column_names = [f'PC{i+1}' for i in range(self.num_PC)]
        y_train = pd.DataFrame(data=electric_data_proj, columns=column_names)

        # Fit model
        model = LinearRegression(fit_intercept=True)
        model.fit(x_train, y_train)
        self.model = model
        self.trained = True

    def predict(self, week, day, year=None, weather=None):
        if weather is None:
            assert False, 'Prediction needs weather data'
        if self.trained == False:
            assert False, "Model not trained"
        if year is None: year = self.year
        x_pred = weather[self.input_features].copy()
        for i, weekday in enumerate(WEEKDAYS[:-1]):
            x_pred[weekday] = (weather['day_of_week'] == i).astype(int)
        for year in self.train_year_list:
            x_pred[f'year{year}'] = (weather['year'] == year).astype(int)
        y_pred = np.array(self.model.predict(x_pred))
        return (y_pred @ self.pca.components_[:self.num_PC] + self.pca.mean_) * np.sqrt(self.metered_variance)

def train_correction_model(param_path, year=2024, train_year=3, num_PC=5, 
                           pca_input_dir = 'data/data_weather_hourly_processed'):

    # Load computed variance and pca
    with open(f'{param_path}/variances.json', 'r') as file:
        variances_global = json.load(file)

    # For each zone: train a model, cal. pred. vs ground truth, save away
    prediction_projs = []
    ground_truth_projs = []
    for zone in ZONES:
        pipeline = PCAGlobalPipeline(zone=zone, year=year, train_year=train_year, \
                                     num_PC=num_PC, pca_input_dir=pca_input_dir)
        pipeline.load_pca(f'{param_path}/pca_global.pkl')
        pipeline.metered_variance = variances_global[zone]
        pipeline.train_model()
        request = {y: slide_week_day(-1, 0, 3, 6)+slide_week_day(0, 1, 3, 6) for y in range(year-1, year-train_year, -1)}
        prediction = pipeline.predict_request(request)
        ground_truth = np.array(get_electric_data(zone, request, \
                                                  daystart=PRED_WEEK_START)[HOURS])
        prediction_proj = pipeline.pca.transform(prediction/np.sqrt(pipeline.metered_variance))[:, :num_PC]
        ground_truth_proj = pipeline.pca.transform(ground_truth/np.sqrt(pipeline.metered_variance))[:, :num_PC]
        # print(prediction[0, :])
        # print(ground_truth[0, :])
        # print(prediction_proj[0, :], ground_truth_proj[0, :])
        # print(((prediction_proj @ pipeline.pca.components_[:pipeline.num_PC] + pipeline.pca.mean_) * \
        #       np.sqrt(pipeline.metered_variance))[0, :])
        # print(((ground_truth_proj @ pipeline.pca.components_[:pipeline.num_PC] + pipeline.pca.mean_) * \
        #       np.sqrt(pipeline.metered_variance))[0, :])
        # print(prediction_proj.shape, ground_truth_proj.shape)
        prediction_projs.append(prediction_proj)
        ground_truth_projs.append(ground_truth_proj)

    # Data preparation
    predictions_np = np.vstack(prediction_projs)
    ground_truths_np = np.vstack(ground_truth_projs)
    column_names = [f'PC{i+1}' for i in range(num_PC)]
    x_train = pd.DataFrame(predictions_np, columns=column_names)
    y_train = pd.DataFrame(ground_truths_np, columns=column_names)
    day_code = np.array([0, 1, 2, 3, 4, 5]*((train_year-1)*len(ZONES)))
    for i in range(5):
        x_train[f'daycode{i}'] = (day_code == i).astype(int)

    # Train the linear model
    print(x_train.head(5))
    print(y_train.head(5))
    model = LinearRegression(fit_intercept=True)
    model.fit(x_train, y_train)

    # Save the linear model
    with open(f'{param_path}/linreg_correction.pkl', 'wb') as file:
        pickle.dump(model, file)

def train_correction_model_by_day(param_path, year=2024, train_year=3, num_PC=5, days=[[3, 0], [4, 0], [5, 0]],
                                  pca_input_dir = 'data/data_weather_hourly_processed'):

    # Load computed variance and pca
    with open(f'{param_path}/variances.json', 'r') as file:
        variances_global = json.load(file)

    # For each zone: train a model, cal. pred. vs ground truth, save away
    prediction_projs = []
    ground_truth_projs = []
    for zone in ZONES:
        pipeline = PCAGlobalPipeline(zone=zone, year=year, train_year=train_year, \
                                     num_PC=num_PC, pca_input_dir=pca_input_dir)
        pipeline.load_pca(f'{param_path}/pca_global.pkl')
        pipeline.metered_variance = variances_global[zone]
        pipeline.train_model()
        request = {y: slide_week_day(-1, 1) for y in range(year-1, year-train_year, -1)}
        prediction = pipeline.predict_request(request)
        ground_truth = np.array(get_electric_data(zone, request, \
                                                  daystart=PRED_WEEK_START)[HOURS])
        prediction_proj = pipeline.pca.transform(prediction/np.sqrt(pipeline.metered_variance))[:, :num_PC]
        ground_truth_proj = pipeline.pca.transform(ground_truth/np.sqrt(pipeline.metered_variance))[:, :num_PC]
        # print(prediction[0, :])
        # print(ground_truth[0, :])
        # print(prediction_proj[0, :], ground_truth_proj[0, :])
        # print(((prediction_proj @ pipeline.pca.components_[:pipeline.num_PC] + pipeline.pca.mean_) * \
        #       np.sqrt(pipeline.metered_variance))[0, :])
        # print(((ground_truth_proj @ pipeline.pca.components_[:pipeline.num_PC] + pipeline.pca.mean_) * \
        #       np.sqrt(pipeline.metered_variance))[0, :])
        # print(prediction_proj.shape, ground_truth_proj.shape)
        prediction_projs.append(prediction_proj)
        ground_truth_projs.append(ground_truth_proj)

    # Data preparation
    predictions_np = np.vstack(prediction_projs)
    ground_truths_np = np.vstack(ground_truth_projs)
    column_names = [f'PC{i+1}' for i in range(num_PC)]
    x_train = pd.DataFrame(predictions_np, columns=column_names)
    y_train = pd.DataFrame(ground_truths_np, columns=column_names)

    for d, w in days:
        print(d, w)

        # Mask which day to correct & extract these days out of data
        day_code = [0]*14
        day_code[w*7+7+((d+1)%7)] = 1
        day_code = np.array(day_code*((train_year-1)*len(ZONES)))
        x_train_temp = x_train[day_code==1].copy()
        y_train_temp = y_train[day_code==1].copy()

        # Train the linear model
        model = LinearRegression(fit_intercept=True)
        model.fit(x_train, y_train)

        # Save the linear model
        with open(f'{param_path}/linreg_correction_{to_string(w)}_{to_string(d)}.pkl', 'wb') as file:
            pickle.dump(model, file)

if __name__ == '__main__':
    request = {year: slide_week_day(-10, 3, daystart=PRED_WEEK_START) for year in [2022, 2023, 2024]}
    param_path = 'params/global_param_2024'
    os.makedirs(param_path, exist_ok=True)
    generate_pca_metered(request=request, param_path=param_path)
    train_correction_model(param_path, year=2024, train_year=3, num_PC=5, 
                           pca_input_dir = 'data/data_weather_hourly_processed')
    train_correction_model_by_day(param_path, year=2024, train_year=3, num_PC=5, days=slide_week_day(-1, 1),
                                  pca_input_dir = 'data/data_weather_hourly_processed')