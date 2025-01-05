
import os
import json
import numpy as np
from maudlin_core.src.model.model import create_model
from maudlin_core.src.lib.framework.maudlin_unit_config import get_current_unit_config
from maudlin_core.src.lib.framework.maudlin import load_maudlin_data
from maudlin_core.src.lib.data_loading.prediction import load_for_prediction
from maudlin_core.src.lib.framework.stage_functions.pre_prediction_function import execute_preprediction_stage
from maudlin_core.src.lib.framework.stage_functions.post_prediction_function import execute_postprediction_stage

# Global variables
maudlin = load_maudlin_data()
config = None

def get_training_dir():
    return os.path.join(maudlin['data-directory'], 'trainings', maudlin['current-unit'])

def get_prediction_dir():
    return os.path.join(maudlin['data-directory'], 'predictions', maudlin['current-unit'])

def load_run_metadata(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return {}

def save_run_metadata(file_path, metadata):
    with open(file_path, 'w') as file:
        json.dump(metadata, file, indent=4)

def setup_training_directory(training_dir):
    if not os.path.exists(training_dir):
        raise ValueError(f"Training directory not found: {training_dir}")

    run_metadata_path = os.path.join(training_dir, 'run_metadata.json')
    run_metadata = load_run_metadata(run_metadata_path)

    current_run_id = run_metadata.get('current_run_id')
    if current_run_id is None:
        raise ValueError(f"current_run_id not found in run_metadata.json: {run_metadata_path}")

    most_recent_run_path = os.path.join(training_dir, f"run_{current_run_id}")
    if not os.path.exists(most_recent_run_path):
        raise ValueError(f"Most recent training run directory not found: {most_recent_run_path}")

    return most_recent_run_path

def setup_prediction_directory(prediction_dir):
    os.makedirs(prediction_dir, exist_ok=True)

    pred_metadata_path = os.path.join(prediction_dir, 'run_metadata.json')
    pred_metadata = load_run_metadata(pred_metadata_path)

    pred_run_id = pred_metadata.get('highest_run_id', 0) + 1
    pred_metadata['highest_run_id'] = pred_run_id
    pred_metadata['current_run_id'] = pred_run_id

    save_run_metadata(pred_metadata_path, pred_metadata)

    prediction_run_dir = os.path.join(prediction_dir, f"run_{pred_run_id}")
    os.makedirs(prediction_run_dir, exist_ok=True)

    return prediction_run_dir

def validate_prediction_file(config):
    data_file = config['data']['prediction_file']
    if not data_file:
        raise ValueError(f"Data file for [{maudlin['current-unit']}] is not set correctly.")
    return data_file

def load_and_preprocess_data(config, data_dir):
    X, y, feature_count, columns = load_for_prediction(config, data_dir)
    return X, y, feature_count, columns

def get_model(config, training_data_dir, feature_count):
    return create_model(config, training_data_dir, feature_count, True)

def make_predictions(config, model, X, y, feature_count, columns, data_dir):
    X_pca, _, _, _ = execute_preprediction_stage(config, data_dir, model, X, y, feature_count, columns)
    predictions = model.predict(X_pca)

    threshold = config.get('prediction', {}).get('threshold', 0.5)
    y_preds = (predictions >= threshold).astype('int')

    execute_postprediction_stage(config, data_dir, model, y_preds, y)


def main():
    # Load configuration dynamically
    config = get_current_unit_config('')
    config['mode'] = 'prediction'

    # Validate and set up directories
    training_dir = get_training_dir()
    prediction_dir = get_prediction_dir()
    training_data_dir = setup_training_directory(training_dir)
    prediction_run_dir = setup_prediction_directory(prediction_dir)

    # Validate prediction file
    validate_prediction_file(config)

    print("\nStarting the script...")
    print("Loading data to use as a basis for prediction, and featurizing it...")

    # Load and preprocess data
    X, y, feature_count, columns = load_and_preprocess_data(config, prediction_run_dir)

    # Load model
    print("Getting the model...")
    model = get_model(config, training_data_dir, feature_count)

    # Make predictions
    make_predictions(config, model, X, y, feature_count, columns, prediction_run_dir)

    print("\nPrediction process completed.")

if __name__ == "__main__":
    main()

