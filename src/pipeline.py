'''
Define the load prediction pipelines.
Each pipeline has a model and the functionalities
to get data, train model, save and load model, test itself in the case of previous years,
and give model prediction.
Each pipeline is basically an algorithm (algo = model + hyperparams).
(hyperparams are sometime input-able in pipeline. We also don't have a lot of them.)
The last ones (named and are most complicated) are the final models used for prediction
'''
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from src.data_loader import get_electric_data, get_weather_data, get_weather_data_hourly, format_request
from src.constants import ZONES, PRED_DAYS, PRED_WEEK_START, HOURS, WEEKDAYS
from src.utils import slide_week_day, calculate_loss, to_string

class ElectricTemplatePipeline:
    '''
    Base Template Pipeline class for the
    Predicting Electricity Load Around Thanksgiving task.
    Class method to be overloaded by child class:   

        _request_train_electric_data(self)
        load_model(self, model_path)
        save_model(self, model_path)
        train_model(self)
        predict(self, week, day, weather=None)

    '''
    def __init__(self, zone='AECO', year=2024):
        self.zone = zone
        self.year = year
        self.trained = False

    def _request_train_electric_data(self):
        raise NotImplementedError

    def _request_train_weather_data(self):
        return self._request_train_electric_data()
    
    def _request_test_electric_data(self):
        if self.year == 2025: return dict()
        return {self.year: PRED_DAYS}
    
    def _request_test_weather_data(self):
        return self._request_test_electric_data()

    def load_model(self, model_path):
        raise NotImplementedError
    
    def save_model(self, model_path):
        raise NotImplementedError
    
    def train_model(self):
        raise NotImplementedError
    
    def test_model(self):
        if self.year == 2025:
            assert False, "No testing available for this year!"
        electric_data = get_electric_data(self.zone, \
                                          self._request_test_electric_data(), \
                                          daystart=PRED_WEEK_START)[HOURS]
        weather_data = get_weather_data(self.zone, \
                                        self._request_test_weather_data(), \
                                        daystart=PRED_WEEK_START)
        day_predicts = format_request(self._request_test_electric_data())
        electric_predict = self.predict(week=day_predicts['relative_week'], \
                                        day=day_predicts['day_of_week'], \
                                        year=day_predicts['year'], weather=weather_data)
        electric_data_np = np.array(electric_data)
        electric_predict_np = np.array(electric_predict)
        peak_days = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 1])
        return calculate_loss(electric_predict_np, electric_data_np, peak_days)

    def predict(self, week, day, year=None, weather=None):
        raise NotImplementedError
    
    def predict_request(self, request): 
        weather_data = get_weather_data(self.zone, request, \
                                        daystart=PRED_WEEK_START)
        day_predicts = format_request(request)
        return self.predict(week=day_predicts['relative_week'], \
                            day=day_predicts['day_of_week'], \
                            year=day_predicts['year'], \
                            weather=weather_data)
    
    def predict_final(self): # In the work
        return [self.predict(week, day) for day, week in PRED_DAYS]
    
class ZeroPipeline(ElectricTemplatePipeline):
    '''
    Simple pipeline that just output a bunch of zeros.
    Nothing to see here.
    '''
    def __init__(self, zone='AECO', year=2024):
        super().__init__(zone, year)
        self.trained = True

    def _request_train_electric_data(self):
        return dict()

    def save_model(self, model_path):
        pass

    def load_model(self, model_path):
        pass
    
    def train_model(self):
        pass
    
    def predict(self, week, day, year=None, weather=None):
        return np.array([[0]*24 for _ in range(len(day))])
    
class BaseMeanPipeline(ElectricTemplatePipeline):
    '''
    Simple pipeline that uses the mean of the previous weeks to predict.
    Very little to see here.
    '''
    def __init__(self, zone='AECO', year=2024, use_week=8):
        super().__init__(zone, year)
        self.use_week = use_week
        self.model = None

    def _request_train_electric_data(self):
        return {self.year: slide_week_day(-2-self.use_week, -2, daystart=PRED_WEEK_START)}
    
    def _request_test_weather_data(self):
        return dict()

    def save_model(self, model_path):
        if self.trained == False:
            assert False, "Model not trained"
        with open(model_path, 'w') as f:
            json.dump(self.model, f, indent=4) 

    def load_model(self, model_path):
        with open(model_path, 'r') as f:
            self.model = json.load(f)
        self.trained = True
    
    def train_model(self):
        electric_data = get_electric_data(self.zone, \
                                          self._request_train_electric_data(), \
                                          daystart=PRED_WEEK_START)
        temp = electric_data.mean()
        self.model = list(temp[[f"H{int(h):02d}" for h in range(24)]])
        self.trained = True
    
    def predict(self, week, day, year=None, weather=None):
        if self.trained == False:
            assert False, "Model not trained"
        return np.array([self.model for _ in day]).copy()
    
class BaseMeanByDayPipeline(ElectricTemplatePipeline):
    '''
    Simple pipeline that uses the mean of the previous weeks to predict.
    Uses the mean of previous mondays to predict monday, etc.
    Little to see here.
    '''
    def __init__(self, zone='AECO', year=2024, use_week=8):
        super().__init__(zone, year)
        self.use_week = use_week
        self.model = [0]*7

    def _request_train_electric_data(self):
        return {self.year: slide_week_day(-2-self.use_week, -2, daystart=PRED_WEEK_START)}
    
    def _request_test_weather_data(self):
        return dict()

    def save_model(self, model_path):
        if self.trained == False:
            assert False, "Model not trained"
        with open(model_path, 'w') as f:
            json.dump(self.model, f, indent=4) 

    def load_model(self, model_path):
        with open(model_path, 'r') as f:
            self.model = json.load(f)
        self.trained = True
    
    def train_model(self):
        electric_data = get_electric_data(self.zone, \
                                          self._request_train_electric_data(), \
                                          daystart=PRED_WEEK_START)
        for day in range(7):
            temp = electric_data[electric_data['day_of_week']==day].mean()
            self.model[day] = list(temp[[f"H{int(h):02d}" for h in range(24)]])
        self.trained = True
    
    def predict(self, week, day, year=None, weather=None):
        if self.trained == False:
            assert False, "Model not trained"
        return np.array([self.model[d] for d in day])

class BasicRegressionPipeline(ElectricTemplatePipeline):
    '''
    Linear Regression Pipleline that predicts the next day based on weather of that day
    and which day it is in the week.
    First proper and non-trivial baseline model.
    '''
    def __init__(self, zone='AECO', year=2024, train_year=1):
        super().__init__(zone, year)
        self.train_year = train_year
        self.train_year_list = None
        self.model = None

    def _request_train_electric_data(self):
        output = {i: slide_week_day(-10, 3, daystart=PRED_WEEK_START) \
                  for i in range(self.year-1, self.year-self.train_year, -1)}
        output[self.year] = slide_week_day(-10, -2, daystart=PRED_WEEK_START)
        return output

    def load_model(self, model_path):
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)
        self.trained = True

    def save_model(self, model_path):
        if self.trained == False:
            assert False, "Model not trained"
        with open(model_path, 'wb') as file:
            pickle.dump(self.model, file)

    def train_model(self):
        electric_data = get_electric_data(self.zone, \
                                          self._request_train_electric_data(), \
                                          daystart=PRED_WEEK_START)
        weather_data = get_weather_data(self.zone, \
                                        self._request_train_weather_data(), \
                                        daystart=PRED_WEEK_START)
        features = ['tmin', 'tmax', 'tavg', 'pres']
        x_train = weather_data[features].copy()
        for i, weekday in enumerate(WEEKDAYS[:-1]):
            x_train[weekday] = (weather_data['day_of_week'] == i).astype(int)
        self.train_year_list = list(weather_data['year'].unique())[:-1]
        for year in self.train_year_list:
            x_train[f'year{year}'] = (weather_data['year'] == year).astype(int)
        y_train = electric_data[HOURS]
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
        x_pred = weather[['tmin', 'tmax', 'tavg', 'pres']].copy()
        for i, weekday in enumerate(WEEKDAYS[:-1]):
            x_pred[weekday] = (weather['day_of_week'] == i).astype(int)
        for year in self.train_year_list:
            x_pred[f'year{year}'] = (weather['year'] == year).astype(int)
        return np.array(self.model.predict(x_pred))
    
class PCARegressionPipeline(ElectricTemplatePipeline):
    '''
    Linear Regression Pipleline that predicts the next day based on weather of that day
    and which day it is in the week.
    Learns a PCA encoding of the daily load, so output some (e.g., 3) PCs instead of 24 hours
    Intuitively: less prone to overfitting, more interpretable, much smaller in size.
    '''
    def __init__(self, zone='AECO', year=2024, train_year=3, train_year_pca=None, num_PC=3):
        super().__init__(zone, year)
        if train_year_pca == None: train_year_pca = train_year
        self.train_year = train_year
        self.train_year_list = None
        self.train_year_pca = train_year_pca
        self.trained_pca = False
        self.model = None
        self.pca = None
        self.num_PC = num_PC
    
    def _request_train_pca_data(self):
        output = {i: slide_week_day(-10, 3, daystart=PRED_WEEK_START) \
                  for i in range(self.year-1, self.year-self.train_year_pca, -1)}
        output[self.year] = slide_week_day(-10, -2, daystart=PRED_WEEK_START)
        return output

    def _request_train_electric_data(self):
        output = {i: slide_week_day(-10, 3, daystart=PRED_WEEK_START) \
                  for i in range(self.year-1, self.year-self.train_year, -1)}
        output[self.year] = slide_week_day(-10, -2, daystart=PRED_WEEK_START)
        return output
    
    def load_pca(self, pca_path):
        with open(pca_path, 'rb') as file:
            self.pca = pickle.load(file)
        self.trained_pca = True

    def load_model(self, model_path):
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)
        self.trained = True

    def save_pca(self, pca_path):
        if self.trained_pca == False:
            assert False, "PCA decompt not trained"
        with open(pca_path, 'wb') as file:
            pickle.dump(self.pca, file)    

    def save_model(self, model_path):
        if self.trained == False:
            assert False, "Model not trained"
        with open(model_path, 'wb') as file:
            pickle.dump(self.model, file)

    def train_pca(self):
        electric_data = get_electric_data(self.zone, \
                                          self._request_train_pca_data(), \
                                          daystart=PRED_WEEK_START)
        pca = PCA()
        pca.fit(np.array(electric_data[HOURS]))
        self.pca = pca
        self.trained_pca = True
        
    def train_model(self):
        # Prepare pca & load data
        if not self.trained_pca: self.train_pca()
        electric_data = get_electric_data(self.zone, \
                                          self._request_train_electric_data(), \
                                          daystart=PRED_WEEK_START)
        weather_data = get_weather_data(self.zone, \
                                        self._request_train_weather_data(), \
                                        daystart=PRED_WEEK_START)
        features = ['tmin', 'tmax', 'tavg', 'pres']

        # Prepare x
        x_train = weather_data[features].copy()
        for i, weekday in enumerate(WEEKDAYS[:-1]):
            x_train[weekday] = (weather_data['day_of_week'] == i).astype(int)
        self.train_year_list = list(weather_data['year'].unique())[:-1]
        for year in self.train_year_list:
            x_train[f'year{year}'] = (weather_data['year'] == year).astype(int)

        # Prepare y
        electric_data_np = np.array(electric_data[HOURS])
        electric_data_proj = self.pca.transform(electric_data_np)[:, :self.num_PC]
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
        x_pred = weather[['tmin', 'tmax', 'tavg', 'pres']].copy()
        for i, weekday in enumerate(WEEKDAYS[:-1]):
            x_pred[weekday] = (weather['day_of_week'] == i).astype(int)
        for year in self.train_year_list:
            x_pred[f'year{year}'] = (weather['year'] == year).astype(int)
        y_pred = np.array(self.model.predict(x_pred))
        return y_pred @ self.pca.components_[:self.num_PC] + self.pca.mean_
    
class PCAWeatherRegressionPipeline(ElectricTemplatePipeline):
    '''
    Linear Regression Pipleline that predicts the next day based on weather of that day
    and which day it is in the week.
    Learns a PCA encoding of the daily load, so output some (e.g., 3) PCs instead of 24 hours
    Instead of taking weather feature (e.g., max temp, min temp, avg temp) as input, take in 
    PCs of them (e.g., temp_PC0, temp_PC1, ...) as input.
    Intuitively: gets more out of weather data, more interpretable.
    Candidate model for prediction.
    '''
    def __init__(self, zone='AECO', year=2024, train_year=3, train_year_pca=None, num_PC=3, \
                 pca_input_dir = 'data/data_weather_hourly_processed'):
        super().__init__(zone, year)
        if train_year_pca == None: train_year_pca = train_year
        self.train_year = train_year
        self.train_year_list = None
        self.train_year_pca = train_year_pca
        self.trained_pca = False
        self.model = None
        self.pca = None
        self.num_PC = num_PC
        self.pca_input_dir = pca_input_dir
        self.pca_inputs = dict()
        self.input_features = []
        self._load_weather_pca()
        self._create_input_features()

    def _load_weather_pca(self):
        with open(f'{self.pca_input_dir}/counts.json', 'r') as f:
            self.counts = json.load(f)
        self.pca_inputs = dict()
        for feature in self.counts.keys():
            with open(f'{self.pca_input_dir}/pcas/{self.zone}_{feature}.pkl', 'rb') as f:
                self.pca_inputs[feature] = pickle.load(f)

    def _create_input_features(self):
        self.input_features = []
        for key, value in self.counts.items():
            for i in range(value):
                self.input_features.append(f'{key}_PC{i}')
    
    def _request_train_pca_data(self):
        output = {i: slide_week_day(-10, 3, daystart=PRED_WEEK_START) \
                  for i in range(self.year-1, self.year-self.train_year_pca, -1)}
        output[self.year] = slide_week_day(-10, -2, daystart=PRED_WEEK_START)
        return output

    def _request_train_electric_data(self):
        output = {i: slide_week_day(-10, 3, daystart=PRED_WEEK_START) \
                  for i in range(self.year-1, self.year-self.train_year, -1)}
        output[self.year] = slide_week_day(-10, -2, daystart=PRED_WEEK_START)
        return output
    
    def load_pca(self, pca_path):
        with open(pca_path, 'rb') as file:
            self.pca = pickle.load(file)
        self.trained_pca = True

    def load_model(self, model_path):
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)
        self.trained = True

    def save_pca(self, pca_path):
        if self.trained_pca == False:
            assert False, "PCA decompt not trained"
        with open(pca_path, 'wb') as file:
            pickle.dump(self.pca, file)    

    def save_model(self, model_path):
        if self.trained == False:
            assert False, "Model not trained"
        with open(model_path, 'wb') as file:
            pickle.dump(self.model, file)

    def train_pca(self):
        electric_data = get_electric_data(self.zone, \
                                          self._request_train_pca_data(), \
                                          daystart=PRED_WEEK_START)
        pca = PCA()
        pca.fit(np.array(electric_data[HOURS]))
        self.pca = pca
        self.trained_pca = True
        
    def train_model(self):
        # Prepare pca & load data
        if not self.trained_pca: self.train_pca()
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
        electric_data_proj = self.pca.transform(electric_data_np)[:, :self.num_PC]
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
        return y_pred @ self.pca.components_[:self.num_PC] + self.pca.mean_
    
    def test_model(self):
        if self.year == 2025:
            assert False, "No testing available for this year!"
        electric_data = get_electric_data(self.zone, \
                                          self._request_test_electric_data(), \
                                          daystart=PRED_WEEK_START)[HOURS]
        weather_data = get_weather_data_hourly(self.zone, \
                                               self._request_test_weather_data(), \
                                               daystart=PRED_WEEK_START, data_dir=self.pca_input_dir)
        day_predicts = format_request(self._request_test_electric_data())
        electric_predict = self.predict(week=day_predicts['relative_week'], \
                                        day=day_predicts['day_of_week'], \
                                        year=day_predicts['year'], weather=weather_data)
        electric_data_np = np.array(electric_data)
        electric_predict_np = np.array(electric_predict)
        peak_days = np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 1])
        return calculate_loss(electric_predict_np, electric_data_np, peak_days)
    
    def predict_request(self, request): 
        weather_data = get_weather_data_hourly(self.zone, request, daystart=PRED_WEEK_START, \
                                               data_dir=self.pca_input_dir)
        day_predicts = format_request(request)
        return self.predict(week=day_predicts['relative_week'], \
                            day=day_predicts['day_of_week'], \
                            year=day_predicts['year'], \
                            weather=weather_data)
    
class Faraday(ElectricTemplatePipeline):
    '''
    First model accounting for holiday effects
    Linear Regression Pipleline that predicts the next day based on weather of that day
    and which day it is in the week.
    Learns a PCA encoding of the daily load, so output some (e.g., 3) PCs instead of 24 hours
    Instead of taking weather feature (e.g., max temp, min temp, avg temp) as input, take in 
    PCs of them (e.g., temp_PC0, temp_PC1, ...) as input.
    Add a 1-0 input covariates to signal and correct for thu, fri and sat of the two weeks (observed hard to learn)
    Intuitively: accout for holiday effects in the (observabled from past years) affected dates.
    Draw backs: weak correction.
    Observed (mainly in 2024) that these days are hard to learn.
    What if the weather was hard to learn, not the holiday effect?
    '''
    def __init__(self, zone='AECO', year=2024, train_year=3, train_year_pca=None, num_PC=5, \
                 pca_input_dir = 'data/data_weather_hourly_processed'):
        super().__init__(zone, year)
        if train_year_pca == None: train_year_pca = train_year
        self.train_year = train_year
        self.train_year_list = None
        self.train_year_pca = train_year_pca
        self.trained_pca = False
        self.model = None
        self.pca = None
        self.num_PC = num_PC
        self.pca_input_dir = pca_input_dir
        self.pca_inputs = dict()
        self.input_features = []
        self._load_weather_pca()
        self._create_input_features()

    def _load_weather_pca(self):
        with open(f'{self.pca_input_dir}/counts.json', 'r') as f:
            self.counts = json.load(f)
        self.pca_inputs = dict()
        for feature in self.counts.keys():
            with open(f'{self.pca_input_dir}/pcas/{self.zone}_{feature}.pkl', 'rb') as f:
                self.pca_inputs[feature] = pickle.load(f)

    def _create_input_features(self):
        self.input_features = []
        for key, value in self.counts.items():
            for i in range(value):
                self.input_features.append(f'{key}_PC{i}')
    
    def _request_train_pca_data(self):
        output = {i: slide_week_day(-10, 3, daystart=PRED_WEEK_START) \
                  for i in range(self.year-1, self.year-self.train_year_pca, -1)}
        output[self.year] = slide_week_day(-10, -2, daystart=PRED_WEEK_START)
        return output

    def _request_train_electric_data(self):
        output = {i: slide_week_day(-10, 3, daystart=PRED_WEEK_START) \
                  for i in range(self.year-1, self.year-self.train_year, -1)}
        output[self.year] = slide_week_day(-10, -2, daystart=PRED_WEEK_START)
        return output
    
    def load_pca(self, pca_path):
        with open(pca_path, 'rb') as file:
            self.pca = pickle.load(file)
        self.trained_pca = True

    def load_model(self, model_path):
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)
        self.trained = True

    def save_pca(self, pca_path):
        if self.trained_pca == False:
            assert False, "PCA decompt not trained"
        with open(pca_path, 'wb') as file:
            pickle.dump(self.pca, file)    

    def save_model(self, model_path):
        if self.trained == False:
            assert False, "Model not trained"
        with open(model_path, 'wb') as file:
            pickle.dump(self.model, file)

    def train_pca(self):
        electric_data = get_electric_data(self.zone, \
                                          self._request_train_pca_data(), \
                                          daystart=PRED_WEEK_START)
        pca = PCA()
        pca.fit(np.array(electric_data[HOURS]))
        self.pca = pca
        self.trained_pca = True
        
    def train_model(self):
        # Prepare pca & load data
        if not self.trained_pca: self.train_pca()
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
        x_train['wkd1'] = ((weather_data['relative_week']==-1) & \
                           (weather_data['day_of_week'].isin([3, 4, 5]))).astype(int)
        x_train['wkd2'] = ((weather_data['relative_week']== 0) & \
                           (weather_data['day_of_week'].isin([3, 4, 5]))).astype(int)

        # Prepare y
        electric_data_np = np.array(electric_data[HOURS])
        electric_data_proj = self.pca.transform(electric_data_np)[:, :self.num_PC]
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
        x_pred['wkd1'] = ((weather['relative_week']==-1) & \
                          (weather['day_of_week'].isin([3, 4, 5]))).astype(int)
        x_pred['wkd2'] = ((weather['relative_week']== 0) & \
                          (weather['day_of_week'].isin([3, 4, 5]))).astype(int)
        y_pred = np.array(self.model.predict(x_pred))
        return y_pred @ self.pca.components_[:self.num_PC] + self.pca.mean_
    
    def test_model(self):
        if self.year == 2025:
            assert False, "No testing available for this year!"
        electric_data = get_electric_data(self.zone, \
                                          self._request_test_electric_data(), \
                                          daystart=PRED_WEEK_START)[HOURS]
        weather_data = get_weather_data_hourly(self.zone, \
                                               self._request_test_weather_data(), \
                                               daystart=PRED_WEEK_START, data_dir=self.pca_input_dir)
        day_predicts = format_request(self._request_test_electric_data())
        electric_predict = self.predict(week=day_predicts['relative_week'], \
                                        day=day_predicts['day_of_week'], \
                                        year=day_predicts['year'], weather=weather_data)
        electric_data_np = np.array(electric_data)
        electric_predict_np = np.array(electric_predict)
        peak_days = np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 1])
        return calculate_loss(electric_predict_np, electric_data_np, peak_days)
    
    def predict_request(self, request): 
        weather_data = get_weather_data_hourly(self.zone, request, daystart=PRED_WEEK_START, \
                                               data_dir=self.pca_input_dir)
        day_predicts = format_request(request)
        return self.predict(week=day_predicts['relative_week'], \
                            day=day_predicts['day_of_week'], \
                            year=day_predicts['year'], \
                            weather=weather_data)
    
class Edison(PCAWeatherRegressionPipeline):
    '''
    Linear Regression Pipleline that predicts the next day based on weather of that day
    and which day it is in the week.
    Learns a PCA encoding of the daily load, so output some (e.g., 3) PCs instead of 24 hours
    Instead of taking weather feature (e.g., max temp, min temp, avg temp) as input, take in 
    PCs of them (e.g., temp_PC0, temp_PC1, ...) as input.
    Learns a correction model that corrects for thu, fri and sat of the two weeks (observed hard to learn).
    One model thats corrects for these 6 days. Takes the predicted PCs and output corrected PCs
    Intuitively: accout for holiday effects in the (observabled from past years) affected dates.
    Correction model uses cross-zone data: Use more data.
    Draw backs: observed (mainly in 2024) that these days are hard to learn. Hardcoded to correct for those 6 days.
    What if the weather was hard to learn, not the holiday effect?
    Correction model is a linear model and corrects for all 6 supposedly-hard-to-learn days: may give weak correction.
    '''
    def __init__(self, param_dir, \
                 zone='AECO', year=2024, train_year=3, train_year_pca=None, num_PC=5, \
                 pca_input_dir = 'data/data_weather_hourly_processed'):
        super().__init__(zone, year, train_year, train_year_pca, num_PC, pca_input_dir)
        if train_year_pca == None: train_year_pca = train_year
        self.correction_model = None
        self.metered_variance = 1.
        self._load_global_params(param_dir)

    def _load_global_params(self, param_dir):
        with open(f'{param_dir}/linreg_correction.pkl', 'rb') as file:
            self.correction_model = pickle.load(file)
        with open(f'{param_dir}/variances.json', 'r') as file:
            self.metered_variance = json.load(file)[self.zone]
        self.load_pca(f'{param_dir}/pca_global.pkl')

    def train_model(self):
        # Prepare pca & load data
        if not self.trained_pca: assert False, 'pca has to be preloaded'
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

    def predict(self, week, day, year=None, weather=None, with_correction=True):
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
        if with_correction:
            day_mask = np.array([-1]*len(weather), dtype='int')
            day_mask += np.array((weather['relative_week'] == -1) & (weather['day_of_week'] == 3)).astype(int)*1
            day_mask += np.array((weather['relative_week'] == -1) & (weather['day_of_week'] == 4)).astype(int)*2
            day_mask += np.array((weather['relative_week'] == -1) & (weather['day_of_week'] == 5)).astype(int)*3
            day_mask += np.array((weather['relative_week'] ==  0) & (weather['day_of_week'] == 3)).astype(int)*4
            day_mask += np.array((weather['relative_week'] ==  0) & (weather['day_of_week'] == 4)).astype(int)*5
            day_mask += np.array((weather['relative_week'] ==  0) & (weather['day_of_week'] == 5)).astype(int)*6
            y_pred_masked = y_pred[day_mask != -1].copy()
            day_mask_positive = day_mask[day_mask != -1].copy()
            if len(y_pred_masked) != 0:
                y_adjust = np.hstack([y_pred_masked,
                                    (day_mask_positive == 0).astype(int)[:, None], 
                                    (day_mask_positive == 1).astype(int)[:, None], 
                                    (day_mask_positive == 2).astype(int)[:, None], 
                                    (day_mask_positive == 3).astype(int)[:, None], 
                                    (day_mask_positive == 4).astype(int)[:, None]])
                y_pred[day_mask != -1] = self.correction_model.predict(y_adjust)
        return (y_pred @ self.pca.components_[:self.num_PC] + self.pca.mean_) * np.sqrt(self.metered_variance)

class Ampere(PCAWeatherRegressionPipeline):
    '''
    Linear Regression Pipleline that predicts the next day based on weather of that day
    and which day it is in the week.
    Learns a PCA encoding of the daily load, so output some (e.g., 3) PCs instead of 24 hours
    Instead of taking weather feature (e.g., max temp, min temp, avg temp) as input, take in 
    PCs of them (e.g., temp_PC0, temp_PC1, ...) as input.
    Exclude data from thankgiving weeks and week before that of all years in training. Instead:
    Learns a correction model that corrects for all 10 prediction days (hence 10 linear correction models)
    Encoorperate cross-zone data
    Intuitively: great idea!
    Draw backs: Little data (only 58 per day if use 2 years, 87 if use 3 years, to fit a
    (num_PC+1)*(num_PC) linear model). Lots of correction (10 pretrained models). Thus, may have a lot of variance.
    Default: num_PC = 5.
    Selected model for prediction. In his time, Andre-Marie Ampere was a professor at Ecole Polytechnique,
    which was the school I went to for my Bachelor degree.
    '''
    def __init__(self, param_dir, correction_days, \
                 zone='AECO', year=2024, train_year=3, train_year_pca=None, num_PC=5, \
                 pca_input_dir = 'data/data_weather_hourly_processed'):
        super().__init__(zone, year, train_year, train_year_pca, num_PC, pca_input_dir)
        if train_year_pca == None: train_year_pca = train_year
        self.correction_models = dict()
        self.correction_days = correction_days
        self.metered_variance = 1.
        self._load_global_params(param_dir)

    def _load_global_params(self, param_dir):
        self.correction_models = dict()
        for d, w in self.correction_days:
            with open(f'{param_dir}/linreg_correction_{to_string(w)}_{d}.pkl', 'rb') as file:
                self.correction_models[(d, w)] = pickle.load(file)
        with open(f'{param_dir}/variances.json', 'r') as file:
            self.metered_variance = json.load(file)[self.zone]
        self.load_pca(f'{param_dir}/pca_global.pkl')

    def _request_train_electric_data(self):
        output = {i: slide_week_day(-10, 2, daystart=PRED_WEEK_START) \
                  for i in range(self.year, self.year-self.train_year_pca, -1)}
        output[self.year] = slide_week_day(-10, -3, daystart=PRED_WEEK_START)
        return output

    def train_model(self):
        # Prepare pca & load data
        if not self.trained_pca: assert False, 'pca has to be preloaded'
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

    def predict(self, week, day, year=None, weather=None, with_correction=True):
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
        if with_correction:
            for d, w in self.correction_days:
                day_mask = np.array([0]*len(weather), dtype='int')
                day_mask += np.array((weather['relative_week'] == w) & (weather['day_of_week'] == d)).astype(int)
                if np.sum(day_mask) != 0:
                    y_pred_masked = y_pred[day_mask != 0].copy()
                    y_pred[day_mask != 0] = self.correction_models[(d, w)].predict(y_pred_masked)
        return (y_pred @ self.pca.components_[:self.num_PC] + self.pca.mean_) * np.sqrt(self.metered_variance)
