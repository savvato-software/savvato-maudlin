import os
import signal
import sys
import json
import numpy as np
import shutil
import argparse
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import TensorBoard

from ..lib.framework.maudlin import load_maudlin_data, get_current_unit_properties
from ..lib.framework.maudlin_unit_config import get_current_unit_config

from ..model.model import create_model, generate_model_file_name

from ..lib.data_loading.training import load_for_training
from ..lib.framework.stage_functions.pre_training_function import execute_pretraining_stage 
from ..lib.framework.stage_functions.post_training_function import execute_posttraining_stage

from ..model.track_best_metric import TrackBestMetric
from ..model.adaptive_learning_rate import AdaptiveLearningRate

from collections import Counter



# BEGIN

# Signal handler for saving the model on Ctrl+C
def signal_handler(signal, frame):
    print(f"\nInterrupt received! Saving the model to current run directory...")
    model.save(MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}. Exiting gracefully.")
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

def initialize_training_run_directory(maudlin):
    # Define paths
    unit_dir = os.path.join(maudlin['data-directory'], 'trainings', maudlin['current-unit'])
    os.makedirs(unit_dir, exist_ok=True)

    # Counter file path
    counter_file = os.path.join(unit_dir, 'run_metadata.json')

    # Initialize metadata
    if os.path.exists(counter_file):
        with open(counter_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {'highest_run_id': 0, 'current_run_id': 0}

    prev_curr_run_id = metadata['current_run_id']

    # Increment the highest run ID
    metadata['highest_run_id'] += 1
    metadata['current_run_id'] = metadata['highest_run_id']

    # Write updated metadata back to the file
    with open(counter_file, 'w') as f:
        json.dump(metadata, f, indent=4)

    # Create directory for the current run
    data_dir = os.path.join(unit_dir, f"run_{metadata['current_run_id']}")
    os.makedirs(data_dir, exist_ok=True)

    return data_dir, metadata['current_run_id'], prev_curr_run_id

# Main function

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with optional training run configuration.")
    parser.add_argument(
        "training_run_path",
        nargs="?",
        default=None,
        help="Path to the training run directory (optional).",
    )
    args = parser.parse_args()

    config = get_current_unit_config(args.training_run_path)
    config['mode'] = 'training'
    maudlin = load_maudlin_data()

    if args.training_run_path:
        config_path = args.training_run_path
    else:
        config_path = maudlin['data-directory'] + get_current_unit_properties(maudlin)['config-path']

    print(f"Using the config at {config_path}")

    data_dir, run_id, parent_run_id = initialize_training_run_directory(maudlin)
    config['run_id'] = run_id
    config['parent_run_id'] = parent_run_id

    MODEL_FILE = generate_model_file_name(config, data_dir)
    DATA_FILE = config['data']['training_file']
    TIMESTEPS = config['timesteps']
    BATCH_SIZE = config['batch_size']
    EPOCHS = config['epochs']
    METRICS = config['metrics']
    LOSS_FUNCTION = config['loss_function']
    USE_CLASS_WEIGHTS = config.get('class_weights')    # useful for non-binary, continuous models; optional

    from keras import backend as K

    # Clear previous Keras session
    K.clear_session()

    if not DATA_FILE:
        raise ValueError(f"Data file for [{maudlin['current-unit']}] is not set correctly.")

    # copy current config file to the run specific data dir
    shutil.copy(config_path, os.path.join(data_dir, "config.yaml"))

    print("\nStarting the script...")

    # Load and preprocess the data
    print("Loading and preprocessing data...")

    X, y, feature_count, columns = load_for_training(config, data_dir)

    # Split data into training, validation, and testing
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

    X_train, y_train, X_test, y_test, X_val, y_val = execute_pretraining_stage(config, data_dir, X_train, y_train, X_test, y_test, X_val, y_val, columns)

    #print("---- back in batch.py -----------------------")
    #print(f"Shape of Xtrain: {X_train.shape}")
    #print(f"Shape of Xval: {X_val.shape}")
    #print(f"Shape of X: {X_test.shape}")
    #print(f"Shape of ytrain: {y_train.shape}")
    #print(f"Shape of yval: {y_val.shape}")

    print(f"Creating the model...")
    model = create_model(config, data_dir, X_train.shape[-1])  # timesteps and feature count

    # Train the model
    try:
        print("Training the model...")

        # Extract class weights
        classes = config.get('classes', [])
        class_weights = {cls['label']: cls['weight'] for cls in classes}

        print("Class Weights:", class_weights)

        # Extract metrics and other settings from the YAML config
        metrics_to_track = config.get("metrics", ["mae"])

        alrconfig = config['training']['adaptive_learning_rate']
        callbacks = [
            AdaptiveLearningRate(metric_name=metrics_to_track[0], patience=alrconfig['patience'], factor=alrconfig['factor'], min_lr=alrconfig['min-lr']),
            TrackBestMetric(metric_names=metrics_to_track, log_dir=data_dir),
            TensorBoard(log_dir= data_dir + "/tensorboard", histogram_freq=1)
        ]

        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weights
        )

        print("\nTraining round completed. Saving the model...")
        model.save(MODEL_FILE)

        execute_posttraining_stage(config, data_dir, model, X_train, y_train, X_test, y_test)

        ## ...and, we're done.

    except KeyboardInterrupt:
        # Save the model if the user interrupts training
        print("\nTraining interrupted. Saving the model...")
        model.save(MODEL_FILE)
        print(f"Model saved to {MODEL_FILE}. Exiting.")


