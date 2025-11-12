import pandas as pd
from datetime import datetime, timedelta
from constants import THANKSGIVING

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

if __name__ == '__main__':

    # Example usage:
    dates = pd.date_range("2024-11-20", "2024-12-10", freq="D")
    df = pd.DataFrame({"date": dates})

    # Example with Sunday as the start of the week (6)
    df_with_weeks = add_relative_week_column(df, "date", year=2024, week_start=6)
    print(df_with_weeks)
