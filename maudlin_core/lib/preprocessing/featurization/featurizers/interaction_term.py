import pandas as pd
from keras.saving import register_keras_serializable

@register_keras_serializable(package="CustomPackage")
def apply(data, columns, method='multiply', new_column_name=None):
    """
    Creates an interaction term between two numeric columns and returns the updated DataFrame.

    Args:
        data (pd.DataFrame): The grouped DataFrame.
        columns (list): A list of two column names to interact.
        method (str): The method of interaction: 'multiply', 'add', 'subtract', 'divide'.
        new_column_name (str, optional): Name for the new interaction column.
                                         Defaults to '<col1>_<method>_<col2>'.

    Returns:
        pd.DataFrame: The updated DataFrame with the interaction term column.
    """

    # Validate the number of columns
    if len(columns) != 2:
        raise ValueError("The 'columns' parameter must contain exactly two column names.")
    
    col1, col2 = columns[0], columns[1]

    # Determine the name of the new interaction column
    if new_column_name is None:
        new_column_name = f"{col1}_{method}_{col2}"

    # Apply the specified interaction method
    if method == 'multiply':
        data[new_column_name] = data[col1] * data[col2]
    elif method == 'add':
        data[new_column_name] = data[col1] + data[col2]
    elif method == 'subtract':
        data[new_column_name] = data[col1] - data[col2]
    elif method == 'divide':
        # Avoid division by zero
        data[new_column_name] = data[col1] / data[col2].replace(0, 1)
    else:
        raise ValueError("Invalid method. Use 'multiply', 'add', 'subtract', or 'divide'.")

    return data

