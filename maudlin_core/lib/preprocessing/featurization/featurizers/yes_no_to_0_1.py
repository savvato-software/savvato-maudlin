from keras.saving import register_keras_serializable
import pandas as pd

@register_keras_serializable(package="CustomPackage")
def apply(data, columns):
    """
    Sets 'yes' values to 1, and 'no' to 0

    Args:
        data (pd.DataFrame): The input DataFrame containing the data.
        columns (list): List of column names to apply the function to.

    Returns:
        pd.DataFrame: Updated DataFrame
    """

    ##
    ## NOTE, if you need this featurizer, it may be best to just edit the source
    ##  file and make the replacements there. I had a problem with perturbations
    ##  and not being able to apply boolean perturbations because the columns it
    ##  was to work on had yes/no in the columns. This would have been taken care
    ##  of using this featurizer, but it doesn't occur till later in the pipeline
    ##  (perturbations are applied after the raw csv loads). The solution was to
    ##  edit the source file, and remove this from the list of featurizers.
    ##

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")

    if not isinstance(columns, list):
        raise ValueError("Columns parameter must be a list of column names.")

    for column in columns:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")

        # Apply the transformation element-wise
        data[column] = data[column].str.lower().map({'yes': 1, 'no': 0})

    return data

