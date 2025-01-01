import numpy as np

def apply(data, columns, lower_percentile=0.01, upper_percentile=0.99):
    for col in columns:
        lower = data[col].quantile(lower_percentile)
        upper = data[col].quantile(upper_percentile)
        data[col] = np.clip(data[col], lower, upper)
    return data

