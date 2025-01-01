from keras.saving import register_keras_serializable
import pandas as pd

@register_keras_serializable(package="CustomPackage")
def apply(data, columns):
    """
    One-hot encodes the specified categorical columns in the input DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data.
        columns (list): List of column names to one-hot encode.

    Returns:
        pd.DataFrame: Updated DataFrame with one-hot encoded columns.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    
    if not isinstance(columns, list):
        raise ValueError("Columns parameter must be a list of column names.")

    # One-hot encode the specified columns
    data = pd.get_dummies(data, columns=columns, drop_first=False, dtype=float)

    return data

