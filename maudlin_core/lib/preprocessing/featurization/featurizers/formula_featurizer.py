
import pandas as pd
import numpy as np
from keras.saving import register_keras_serializable

@register_keras_serializable(package="CustomPackage")
def apply(orig_data, features):
    """
    Formula-Based Featurizer.

    Args:
        data (pd.DataFrame): Input data.
        features (list): List of dictionaries, each containing 'name' and 'formula'.

    Returns:
        pd.DataFrame: DataFrame with new features added.
    """

    # Copy input data to avoid modifying it directly
    data = orig_data.copy()

    # Iterate through each feature definition

    for feature in features:
        # Validate feature definition
        name = feature.get('name')  # Name of the new column
        formula = feature.get('formula')  # Formula for generating values

        if not name or not formula:
            raise ValueError(f"Invalid feature definition: {feature}")

        try:
            for feature in features:
                # Get the formula and preprocess
                formula = feature['formula']

                # Evaluate with safer eval
                data[name] = eval(formula, {"np": np, "pd": pd}, {"data": data})

            # Ensure boolean outputs are converted to integers
            if data[name].dtype == 'bool':
                data[name] = data[name].astype(int)

        except Exception as e:
            raise ValueError(f"Error processing formula for {name}: {e}")

    # Return the transformed dataset
    return data

