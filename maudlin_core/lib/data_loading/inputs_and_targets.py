import numpy as np

from ..framework.maudlin import load_maudlin_data, get_unit_function_path

import matplotlib.pyplot as plt

from ..savvato_python_functions import load_function_from_file

maudlin = load_maudlin_data()

def get_inputs(config, featurized_df):
    timesteps = config['timesteps']

    # Extract time periods from the YAML
    ytimeperiods = config.get("y-timeperiods", [])
    if not ytimeperiods:
        raise ValueError("No 'ytimeperiods' specified in the configuration.")

    target_period = max(ytimeperiods)  # Use the longest target period for single output

    print("Calculating inputs, Xs")

    # Create inputs (X)
    X = []

    input_function_path = get_unit_function_path(maudlin, 'input-function')
    input_function = load_function_from_file(input_function_path, "apply")
    
    for i in range(len(featurized_df)):
        input = input_function(config, featurized_df, i, timesteps, target_period)
        X.append(input)

    # Convert to numpy arrays
    X = np.array(X)  # [num_samples, timesteps, feature_count]
    X.astype('float32')
    print(f"Shape of X: {X.shape}")  # Expect [num_samples, timesteps, feature_count]

    # Calculate dynamic feature count 
    dynamic_feature_count = X.shape[-1]

    return X, dynamic_feature_count


def get_targets(config, full_df):
    timesteps = config['timesteps']

    # Extract time periods from the YAML
    ytimeperiods = config.get("y-timeperiods", [])
    if not ytimeperiods:
        raise ValueError("No 'ytimeperiods' specified in the configuration.")

    target_period = max(ytimeperiods)  # Use the longest target period for single output

    target_function_path = get_unit_function_path(maudlin, 'target-function')
    target_function = load_function_from_file(target_function_path, "apply")

    print("Calculating targets, Ys")

    # Create outputs (Y)
    Y = []

    for i in range(len(full_df)):
        target = target_function(config, full_df, i, timesteps, target_period)
        Y.append(target)


    # Convert to numpy arrays
    Y = np.array(Y)  # [num_samples, 1]
    Y.astype('float32')

    # Check shapes
    print(f"Shape of Y: {Y.shape}")  # Expect [num_samples, 1]

    # **Data Exploration: Targets**
    ##
    ## TODO This should go somewhere else..... Maybe the pre training function of the unit.. 
    ##   A good time would be when I get a multiclassification model
    ##
    ##
    if len(Y.shape) > 1:
        class_indices = np.argmax(Y, axis=1)
        class_distribution = dict(zip(*np.unique(class_indices, return_counts=True)))
        print(f"Target Distribution: {class_distribution}")

        plt.hist(class_indices, bins=len(config['data_bin_labels']), edgecolor="black")
        plt.xticks(list(range(len(config['data_bin_labels']))), labels=config['data_bin_labels'])
        plt.title("Target Distribution")
        plt.show()

    return Y

