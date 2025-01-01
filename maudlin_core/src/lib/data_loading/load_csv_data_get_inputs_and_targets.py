import importlib.util
import os
import sys
import random
from datetime import datetime

from pathlib import Path

from .inputs_and_targets import get_inputs, get_targets 
from ..framework.maudlin import load_maudlin_data
from ..preprocessing.featurization.featurization import featurize
from ..preprocessing.perturbation import apply_perturbations
from ..savvato_python_functions import read_csv_into_dataframe



maudlin = load_maudlin_data()


def load_csv_data_get_inputs_and_targets(config, csv_data_file_path, generate_targets=True):

    EXTRA_RECORDS_RANDOM_QUANTITY = config.get('extra_records_random_quantity', 0)

    # Load data
    records_to_read = config.get('record_count', 100000)
    multiplier = None
    random_num_up_to_multiplier = 0
    if EXTRA_RECORDS_RANDOM_QUANTITY:
        multiplier = EXTRA_RECORDS_RANDOM_QUANTITY
        random_num_up_to_multiplier = random.uniform(0, multiplier)

    data = read_csv_into_dataframe(csv_data_file_path, int(records_to_read + (records_to_read * random_num_up_to_multiplier)))
    data.columns = config['data']['columns']['csv']

    ##
    ##
    ## TODO: It would be good to check the data first to see if it triggers any known pitfalls, or potential best practices..
    ##  For instance, if a dataset with a column like occupation was passed in, and it has a single defined column but when
    ##  one hot encoded it would produce many columns, suggest using target encoding. that kind of thing..
    ##
    ##  or if both one_hot and target encoding were configured for the same columns...
    ##  or if target_function is run before yes_no_to_0_1.. flag it
    ##

    df = apply_perturbations(config, data)

    # turn csv based dataframe into feature-rich dataframe with scaled values
    sdf = featurize(config, df)

    y = None
    
    if generate_targets:
        y = get_targets(config, sdf) 
    

    ## After featurization we may have columns that were useful for generating features, but shouldn't necessarily be trained on
    ##  so here, we take a subset of the current set of columns, before generating inputs (Xs)

    subset_sdf = None

    subset_columns = config['data']['columns']['final']

    if subset_columns:
        # Initialize a new list to store the final matched column names
        matched_columns = []
        
        # Iterate over the featurized column names in sdf
        for column in sdf.columns:
            # Check if the column starts with any name in the config's final list
            for original_col in subset_columns:
                if column.startswith(f"{original_col}_") or column == original_col:
                    matched_columns.append(column)
                    break  # Avoid duplicate matches for the same prefix
        
        # Update the DataFrame to only include the matched columns
        if matched_columns:
            subset_sdf = sdf[matched_columns]
        else:
            raise ValueError("No columns from config['data']['columns']['final'] matched the DataFrame columns.")

    X, feature_count = get_inputs(config, subset_sdf)

    if generate_targets:
        return X, y, feature_count, subset_sdf.columns.tolist()
    else:
        return X, None, feature_count, subset_sdf.columns.tolist()


