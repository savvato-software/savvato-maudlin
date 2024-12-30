import os
import pandas as pd

from maudlin import load_maudlin_data

from ...savvato_python_functions import load_function_from_file


maudlin = load_maudlin_data()

def featurize(config, data):
    """
    Featurize the given data based on the current unit configuration.

    Args:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The processed DataFrame with features added.
    """
    # Create a copy of the data to avoid modifying the original
    features = data.copy()

    # Access columns configuration
    columns_config = config.get('data', {}).get('columns', {})

    # Handle group_by configuration
    group_by_config = columns_config.get('group_by', {})
    if group_by_config:
        column_name = group_by_config.get('column')
        column_type = group_by_config.get('type')
        aggregations = group_by_config.get('aggregations', [])
        
        # Convert column to specified type
        if column_type == 'date' and column_name in features.columns:
            features[column_name] = pd.to_datetime(features[column_name])
        
        # Build aggregation dictionary
        agg_dict = {
            agg['column']: agg['agg'] for agg in aggregations
        }
        
        # Perform grouping and aggregation
        if column_name in features.columns:
            features = features.groupby(features[column_name].dt.date).agg(agg_dict).reset_index()

    features_config = config.get("data", {}).get("features", [])
    feature_function_map = create_feature_function_map(config)
    return add_features(features_config, feature_function_map, features)


def add_features(features_config, feature_function_map, dataframe):
    """
    Add features to the DataFrame using the provided feature function map.

    Args:
        features_config (list): Configuration for features to be added.
        feature_function_map (dict): Map of feature names to their respective functions.
        dataframe (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with added features.
    """
    for feature in features_config:
        feature_name = feature.get('name')
        params = feature.get('params', {})
        feature_function = feature_function_map.get(feature_name)
        if feature_function:
            dataframe = feature_function(dataframe, **params)

            ## TODO: ensure that the feature_function is returning a dataframe of the expected column length


    return dataframe


def create_feature_function_map(config):
    """
    Create a map of feature names to functions that add those features to a DataFrame.

    Args:
        config (dict): The YAML configuration dictionary containing feature definitions.
        base_path (str): Base path where the feature files are located.

    Returns:
        dict: A dictionary where keys are feature names and values are functions.
    """
    feature_map = {}
    features = config.get('data', {}).get('features', [])
    
    for feature in features:
        feature_name = feature.get('name')
        if not feature_name:
            continue
        
        # Construct the path to the feature file
        base_path = "~/src/savvato-maudlin-lib/featurizing_functions"
        function_file_path = os.path.expanduser(f"{base_path}/{feature_name}.py")
        
        # Load the function
        try:
            feature_function = load_function_from_file(function_file_path, "apply")
            feature_map[feature_name] = feature_function
        except Exception as e:
            print(f"Error loading function for feature '{feature_name}': {e}")
    
    return feature_map



