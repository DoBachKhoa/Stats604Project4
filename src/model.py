import numpy as np
import pandas as pd

class ElectricModel:
    def __init__(self):
        pass

    def load_model(self, model_path=None):
        raise NotImplementedError
    
    def save_model(self, model_path=None):
        raise NotImplementedError
    
    def train(self, data_load, data_weather):
        raise NotImplementedError
    
    def predict(self, day, month):
        raise NotImplementedError
    
class BaseModel(ElectricModel):
    def __init__(self):
        super().__init__()

class RegressionModel(ElectricModel):
    def __init__(self):
        super().__init__()