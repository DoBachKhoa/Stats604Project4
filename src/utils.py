import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.constants import THANKSGIVING

def slide_week_day(week1, week2, day1=None, day2=None, daystart=0):
    if day1 == None: day1 = daystart
    if day2 == None: day2 = daystart
    output = [[day1, week1]]
    d, w = day1, week1
    while True:
        d += 1
        if d == 7: d = 0
        if d == daystart: w += 1
        if w == week2 and d == day2: return output
        else: output.append([d, w])

def add_relative_week_column(df, date_col, year, week_start=0):
    """
    Adds two columns to the given DataFrame:
      - 'relative_week': Week number relative to Thanksgiving week 2024 (week 0).
      - 'day_of_week': Day of week as integer (0=Monday, 6=Sunday).

    PARAMETERS
    ----------
      df (pd.DataFrame): DataFrame with a date column.
      date_col (str): Name of the date column.
      year (int): Year of the df.
      week_start (int): Day the week starts on (0=Monday, 6=Sunday).
    """

    # Ensure date column is datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Thanksgiving date
    thanksgiving = datetime(year, 11, THANKSGIVING[year])

    # Calculate the weekday offset to align with custom week start
    offset = (thanksgiving.weekday() - week_start) % 7
    week0_start = thanksgiving - timedelta(days=offset)

    # Compute the number of weeks difference from week 0
    df["relative_week"] = ((df[date_col] - week0_start).dt.days // 7)

    # Add day of week (0=Monday, 6=Sunday)
    df["day_of_week"] = df[date_col].dt.weekday

    return df

def select_argmax_window3(hours):
    if len(hours.shape) == 1:
        hours_ = hours.copy()
        hours_[:-1] += hours[1:]
        hours_[1:] += hours[:-1]
        return np.argmax(hours_)
    elif len(hours.shape) == 2:
        hours_ = hours.copy()
        hours_[:, :-1] += hours[:, 1:]
        hours_[:, 1:] += hours[:, :-1]
        return np.argmax(hours_, axis=1)
    else:
        raise ValueError("Input `hours` has inappropriate dimension")
    
def calculate_loss(y_predict, y_truth, peak_days=None):
    if peak_days is None: peak_days = np.array([0]*10) # Predict non-peak for all days
    if (not isinstance(y_predict, np.ndarray)) or (not isinstance(y_truth, np.ndarray)):
        raise ValueError('Inputs need to be of type numpy.array')
    if (y_predict.shape != (10, 24)) or (y_truth.shape != (10, 24)):
        raise ValueError(f'Got shape {y_predict.shape} and {y_truth.shape}, but have to be (10, 24)')
    argmax_predict = select_argmax_window3(y_predict)
    argmax_truth = np.argmax(y_truth, axis=1)
    day_maxes = np.max(y_truth, axis=1)
    day_indices = np.argsort(day_maxes)
    a, b = day_indices[-1], day_indices[-2]
    lost1 = np.sqrt(np.sum((y_predict-y_truth)**2/24/10))
    lost2 = (np.abs(argmax_predict-argmax_truth) > 1.5).sum()
    lost3 = 0
    for i in range(10):
        if i in [a, b] and peak_days[i] == 0: lost3 += 4
        if not i in [a, b] and peak_days[i] == 1: lost3 += 1
    lost_ratios = list(np.sum((y_predict-y_truth)**2/24/10, axis=1)/np.sum((y_predict-y_truth)**2/24/10))
    output = {'lost1': lost1, 'lost2': lost2, 'lost3': lost3, '1stHigh': a, '2ndHigh': b, 'lost_ratios': lost_ratios}
    return output
    

if __name__ == '__main__':

    # Example usage:
    dates = pd.date_range("2024-11-20", "2024-12-10", freq="D")
    df = pd.DataFrame({"date": dates})

    # Example with Sunday as the start of the week (6)
    df_with_weeks = add_relative_week_column(df, "date", year=2024, week_start=6)
    print(df_with_weeks)

    # Example if select_argmax_window3
    print()
    print("Argmax window selection test")
    array1 = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    array2 = np.array([1, 2, 5, 3, 7, 6, 9, 3])
    array3 = np.array([[1, 2, 5, 3, 7, 6, 9, 3],
                       [4, 7, 6, 8, 2, 3, 4, 9],
                       [8, 8, 8, 2, 9, 9, 3, 1]]) 
    print(select_argmax_window3(array1))                                                                                                                                                                                 
    print(select_argmax_window3(array2))
    print(select_argmax_window3(array3))
