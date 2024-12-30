from keras.saving import register_keras_serializable
import pandas as pd
import numpy as np

@register_keras_serializable(package="CustomPackage")
def apply(data, columns):
    """
    Applies Min-Max scaling to specified columns in the DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data.
        columns (list): List of column names to apply Min-Max scaling.

    Returns:
        pd.DataFrame: Updated DataFrame with scaled columns.
    """

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")

    if not isinstance(columns, list):
        raise ValueError("Columns parameter must be a list of column names.")

    for column in columns:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")
        
        # Calculate min and max values
        col_min = data[column].min()
        col_max = data[column].max()
        
        if col_max == col_min:
            # Avoid division by zero
            data[column] = 0.0
        else:
            # Apply Min-Max scaling
            data[column] = (data[column] - col_min) / (col_max - col_min)

    #print(" ---- MIN MAX scale ---- ")
    #print(len(data))
    #print(data)

    return data

