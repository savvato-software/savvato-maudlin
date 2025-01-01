import pandas as pd
from keras.saving import register_keras_serializable

@register_keras_serializable(package="CustomPackage")
def apply(data, column, bins, labels, new_column_name=None):
    """
    Bins a numeric column into categorical ranges and returns the updated DataFrame.

    Args:
        data (pd.DataFrame): The grouped DataFrame.
        column (str): The name of the column to bin.
        bins (list): List of bin edges.
        labels (list): List of labels for each bin.
        new_column_name (str, optional): Name for the new binned column. Defaults to '<column>_binned'.

    Returns:
        pd.DataFrame: The updated DataFrame with the binned column.
    """
    
    # Ensure bins and labels are compatible
    if len(bins) - 1 != len(labels):
        raise ValueError("The number of bins must be one more than the number of labels.")
    
    # Determine the name of the new column
    if new_column_name is None:
        new_column_name = f"{column}_binned"

    # Bin the column using pd.cut
    data[new_column_name] = pd.cut(data[column], bins=bins, labels=labels, include_lowest=True)

    data.drop(columns=[column], inplace=True)

    return data

