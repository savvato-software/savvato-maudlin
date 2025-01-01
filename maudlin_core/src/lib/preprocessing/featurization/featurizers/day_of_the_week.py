import pandas as pd

def apply(data):
    # Convert the "Date" column to datetime if it isn't already
    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")
    
    # Map the day of the week (0 = Sunday, ..., 6 = Saturday)
    data["day_of_the_week"] = (data["date"].dt.dayofweek + 1) % 7
    
    return data


