import json
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from data_loader import get_electric_data, get_weather_data, format_request
from constants import ZONES, PRED_DAYS, PRED_WEEK_START, HOURS
from utils import slide_week_day, select_argmax_window3, calculate_loss

class ElectricPipeline:
    '''
    Base Pipeline class for the
    Predicting Electricity Load Around Thanksgiving task.
    Class method to be overloaded by child class:   

        _request_train_electric_data(self)
        load_model(self, model_path=None)
        save_model(self, model_path=None)
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

    def load_model(self, model_path=None):
        raise NotImplementedError
    
    def save_model(self, model_path=None):
        raise NotImplementedError
    
    def train_model(self):
        raise NotImplementedError
    
    def test_model(self):
        if self.year == 2025:
            assert False, "No testing available for this year!"
        electric_data = get_electric_data(self.zone, \
                                          self._request_test_electric_data(), \
                                          daystart=PRED_WEEK_START)
        weather_data = get_weather_data(self.zone, \
                                        self._request_test_weather_data(), \
                                        daystart=PRED_WEEK_START)
        day_predicts = format_request(self._request_test_electric_data())
        electric_predict = self.predict(week=day_predicts['relative_week'], \
                                        day=day_predicts['day_of_week'], \
                                        year=day_predicts['day'], weather=weather_data)
        electric_data_np = np.array(electric_data)
        electric_predict_np = np.array(electric_predict)
        peak_days = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
        return calculate_loss(electric_predict_np, electric_data_np, peak_days)

    def predict(self, week, day, year=None, weather=None):
        raise NotImplementedError
    
    def predict_final(self): # In the work
        return [self.predict(week, day) for day, week in PRED_DAYS]
    
class ZeroPipeline(ElectricPipeline):
    def __init__(self, zone='AECO', year=2024):
        super().__init__(zone, year)
        self.trained = True

    def _request_train_electric_data(self):
        return dict()

    def save_model(self, model_path=None):
        pass

    def load_model(self, model_path=None):
        pass
    
    def train_model(self):
        pass
    
    def predict(self, day, month, year=None, weather=None):
        return np.array([0]*24)
    
class BaseMeanPipeline(ElectricPipeline):
    def __init__(self, zone='AECO', year=2024, use_week=6):
        super().__init__(zone, year)
        self.use_week = use_week
        self.model = [0]*7

    def _request_train_electric_data(self):
        return {self.year: slide_week_day(-4-self.use_week, -4, daystart=PRED_WEEK_START)}
    
    def _request_test_weather_data(self):
        return dict()

    def save_model(self, model_path=None):
        if self.trained == False:
            assert False, "Model not trained"
        with open(model_path, 'w') as f:
            json.dump(self.model, f, indent=4) 

    def load_model(self, model_path=None):
        with open(model_path, 'r') as f:
            self.model = json.load(f)
        self.trained = True
    
    def train_model(self):
        electric_data = get_electric_data(self.zone, \
                                          self._request_train_electric_data(), \
                                          daystart=PRED_WEEK_START)
        for day in range(7):
            temp = electric_data[electric_data['week_day']==day].mean()
            self.model[day] = list(temp[[f"H{int(h):02d}" for h in range(24)]])
        self.trained = True
    
    def predict(self, week, day, year=None, weather=None):
        return np.array(self.model[day])

class RegressionPipeline(ElectricPipeline):
    def __init__(self, zone='AECO', year=2024, train_year=1):
        super().__init__(zone, year)
        self.train_year = train_year
        self.model = None

    def _request_train_electric_data(self):
        return {i: slide_week_day(-10, -4, daystart=PRED_WEEK_START) \
                for i in range(self.year, self.year-self.train_year, -1)}

    def load_model(self, model_path=None):
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)
        self.trained = True

    def save_model(self, model_path=None):
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
        features = ['tmin', 'tmax', 'tavg', 'pres', 'day_of_week']
        if self.train_year > 1: features.append('year')
        x_train = weather_data[features]
        y_train = electric_data[HOURS]
        model = LinearRegression(fit_intercept=True)
        model.fit(x_train, y_train)
        self.model = model
        self.trained = True

    def predict(self, week, day, year=None, weather=None):
        if year is None: year = self.year
        x_pred = weather[['tmin', 'tmax', 'tavg', 'pres']]
        x_pred['day_of_week'] = day
        if self.train_year > 1: x_pred['year'] = year
        return np.array(self.model.predict(x_pred))

class AdditivePipeline(ElectricPipeline):
    def __init__(self, zone='AECO', year=2024):
        super().__init__(zone, year)

    def _request_train_electric_data(self): # In the work
        pass

    def load_model(self, model_path=None): # In the work
        pass
        self.trained = True

    def save_model(self, model_path=None): # In the work
        if self.trained == False:
            assert False, "Model not trained"
        pass

    def train_model(self): # In the work
        pass
        self.trained = True

    def predict(self, week, day, year=None, weather=None): # In the work
        if year is None: year = self.year
        pass
