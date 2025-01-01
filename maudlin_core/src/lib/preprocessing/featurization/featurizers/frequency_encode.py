import pandas as pd
from keras.saving import register_keras_serializable

@register_keras_serializable(package="CustomPackage")
def apply(data, columns):
    """
    Frequency Encoding Featurizer.

    Args:
        data (pd.DataFrame): Input DataFrame.
        columns (list): List of categorical columns to encode.

    Returns:
        pd.DataFrame: DataFrame with frequency-encoded values for the specified columns.
    """

    # Step 1: Create a copy of the data to avoid modifying the original DataFrame
    encoded_data = data.copy()

    # Step 2: Compute frequency encoding for each specified column
    for column in columns:
        # Calculate frequency of each category
        freq_map = data[column].value_counts(normalize=True).to_dict()
        # Map frequency values back to the column
        encoded_data[column + '_freq'] = data[column].map(freq_map)

    return encoded_data

