import os
import json
import numpy as np

# Main function
from model import create_model
from maudlin_unit_config import get_current_unit_config
from maudlin import load_maudlin_data, get_unit_function_path
from savvato_python_functions.savvato_python_functions import load_function_from_file
from data_loading_function_prediction import load_for_prediction
from pre_prediction_function import execute_preprediction_stage
from post_prediction_function import execute_postprediction_stage


maudlin = load_maudlin_data()
config = get_current_unit_config('')
TIMESTEPS = config['timesteps']
DATA_FILE = config['data']['prediction_file']


def setup_training_and_prediction_dirs():
    # Set up the training directory
    training_dir = maudlin['data-directory'] + "/trainings/" + maudlin['current-unit']
    if not os.path.exists(training_dir):
        raise ValueError(f"Training directory not found: {training_dir}")

    # Look for the run_metadata.json file
    run_metadata_path = os.path.join(training_dir, 'run_metadata.json')
    if not os.path.exists(run_metadata_path):
        raise ValueError(f"run_metadata.json not found in training directory: {training_dir}")

    # Read the current training run ID
    with open(run_metadata_path, 'r') as file:
        run_metadata = json.load(file)
        current_run_id = run_metadata.get('current_run_id')

    if current_run_id is None:
        raise ValueError(f"current_run_id not found in run_metadata.json: {run_metadata_path}")

    # Set the most recent training run path
    most_recent_training_run = os.path.join(training_dir, f"run_{current_run_id}")
    if not os.path.exists(most_recent_training_run):
        raise ValueError(f"Most recent training run directory not found: {most_recent_training_run}")

    # Save the path in the config
    config['training_run_path'] = most_recent_training_run

    # Set up the prediction directory
    prediction_dir = maudlin['data-directory'] + "/predictions/" + maudlin['current-unit']
    os.makedirs(prediction_dir, exist_ok=True)

    # Manage prediction metadata
    pred_metadata_path = os.path.join(prediction_dir, 'run_metadata.json')
    if os.path.exists(pred_metadata_path):
        with open(pred_metadata_path, 'r') as file:
            pred_metadata = json.load(file)
            highest_run_id = pred_metadata.get('highest_run_id', 0)
            pred_run_id = highest_run_id + 1
    else:
        pred_run_id = 1

    # Update and save the prediction metadata
    pred_metadata = {
        'highest_run_id': pred_run_id,
        'current_run_id': pred_run_id
    }

    with open(pred_metadata_path, 'w') as file:
        json.dump(pred_metadata, file, indent=4)

    # Create the prediction run directory
    prediction_run_dir = os.path.join(prediction_dir, f"run_{pred_run_id}")
    os.makedirs(prediction_run_dir, exist_ok=True)

    # Return the prediction directory
    return prediction_run_dir

if __name__ == "__main__":

    config['mode'] = 'prediction'

    if not DATA_FILE:
        raise ValueError(f"Data file for [{maudlin['current-unit']}] is not set correctly.")

    data_dir = setup_training_and_prediction_dirs()

    print("\nStarting the script...")

    print("Loading data to use as a basis for prediction, and featurizing it...")

    # Load and preprocess the data
    X, y, feature_count, columns = load_for_prediction(config, data_dir)

    # Load or create the model
    print("Getting the model...")

    model = create_model(config, data_dir, feature_count, True)

    X_pca, _, _, _ = execute_preprediction_stage(config, data_dir, model, X, y, feature_count, columns)

    predictions = model.predict(X_pca)

    threshold = config.get('prediction', {}).get('threshold', 0.5)

    y_preds = (predictions >= threshold).astype('int')

    execute_postprediction_stage(config, data_dir, model, y_preds, y)

    ## ...and, we're done.
