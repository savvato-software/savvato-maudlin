import pandas as pd
from keras.saving import register_keras_serializable

@register_keras_serializable(package="CustomPackage")
def apply(data, columns, target_column='y', smoothing=0.0):
    """
    Multi-Column Target Encoding Featurizer.
    
    Args:
        data (pd.DataFrame): The grouped DataFrame.
        columns (list): List of categorical columns to encode.
        target_column (str): Name of the target column ('y' by default).
        smoothing (float): Smoothing factor to reduce overfitting for low sample sizes.
    
    Returns:
        pd.DataFrame: DataFrame with target-encoded values for the specified columns.
    """

    # Step 1: Compute global mean of the target
    global_mean = data[target_column].mean()

    # Step 2: Apply target encoding to each column in the list
    encoded_data = data.copy()
    for column in columns:
        # Group by column and calculate target mean and count
        target_means = data.groupby(column)[target_column].agg(['mean', 'count'])
        target_means['smoothed_mean'] = (
            (target_means['mean'] * target_means['count'] + global_mean * smoothing) /
            (target_means['count'] + smoothing)
        )
        
        # Map smoothed means back to the data
        encoding_map = target_means['smoothed_mean']
        encoded_data[column] = data[column].map(encoding_map).fillna(global_mean)

    return encoded_data

