import os
import signal
import sys
import json
import yaml
import numpy as np
import shutil
import argparse
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard

from maudlin_core.src.lib.framework.maudlin import load_maudlin_data, get_current_unit_properties
from maudlin_core.src.lib.framework.maudlin_unit_config import get_current_unit_config
from maudlin_core.src.model.model import create_model, generate_model_file_name
from maudlin_core.src.lib.data_loading.training import load_for_training
from maudlin_core.src.lib.framework.stage_functions.pre_training_function import execute_pretraining_stage
from maudlin_core.src.lib.framework.stage_functions.post_training_function import execute_posttraining_stage
from maudlin_core.src.model.track_best_metric import TrackBestMetric
from maudlin_core.src.model.adaptive_learning_rate import AdaptiveLearningRate

from maudlin_core.src.common.data_preparation_manager import DataPreparationManager


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

def setup_signal_handler(training_manager):
    """Set up Ctrl+C handler for graceful exit"""
    def signal_handler(sig, frame):
        print("\nInterrupt received! Saving the model to current run directory...")
        if training_manager.save_model():
            print(f"Model saved to {training_manager.model_file}. Exiting gracefully.")
        else:
            print("No model to save!")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

def run_batch_training(cli_args=None):
    """Main entry point, now orchestrating the training process"""
    # Parse CLI args
    parser = argparse.ArgumentParser(description="Train a model with optional training run configuration.")
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        help="Number of epochs to train the model for."
    )
    parser.add_argument(
        "training_run_path",
        nargs="?",
        default=None,
        help="Path to the training run directory (optional).",
    )
    args = parser.parse_args(cli_args) if cli_args is not None else parser.parse_args()

    # Load configuration
    if not args.training_run_path:
        raise ValueError("training_run_path is required")

    config = get_current_unit_config(args.training_run_path)
    config['mode'] = 'training'
    maudlin = load_maudlin_data()

    # Setup directories
    config_path = args.training_run_path # or maudlin['data-directory'] + get_current_unit_properties(maudlin)['config-path']
    data_dir, run_id, parent_run_id = initialize_training_run_directory(maudlin)
    config['run_id'] = run_id
    config['parent_run_id'] = parent_run_id

    """Configure training callbacks"""
    metrics_to_track = self.config.get("metrics", ["mae"])
    alrconfig = self.config['training']['adaptive_learning_rate']

    callbacks = [
        AdaptiveLearningRate(
            metric_name=metrics_to_track[0],
            patience=alrconfig['patience'],
            factor=alrconfig['factor'],
            min_lr=alrconfig['min-lr']
        ),
        TrackBestMetric(metric_names=metrics_to_track, log_dir=self.data_dir),
        TensorBoard(log_dir=self.data_dir + "/tensorboard", histogram_freq=1)
    ]

    # Initialize training manager
    training_manager = DataPreparationManager(config, data_dir)
    setup_signal_handler(training_manager)

    # Copy config file
    _, curr_run_config_path = training_manager.copy_config_file(config_path + "/config.yaml")

    # Update epochs in the config if specified
    if hasattr(args, 'epochs') and args.epochs is not None:
        config['epochs'] = args.epochs
        with open(curr_run_config_path, 'w') as f:
            yaml.safe_dump(config, f)
        print(f"Updated epochs to {args.epochs} in {curr_run_config_path}")


    try:
        # Load and prepare data
        X_train, y_train, X_test, y_test, X_val, y_val = training_manager.load_and_prepare_data()
        
        # Setup and train model
        training_manager.setup_model(X_train.shape[-1])
        training_manager.train_model(callbacks, X_train, y_train, X_val, y_val)
        
        # Save model and execute post-training
        training_manager.save_model()
        execute_posttraining_stage(config, data_dir, training_manager.model, X_train, y_train, X_test, y_test)

    except KeyboardInterrupt:
        training_manager.save_model()
        print(f"Training interrupted. Model saved to {training_manager.model_file}. Exiting.")

    # append date time
    dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"run_batch_training completed successfully at {dt}!")

if __name__ == "__main__":
    run_batch_training()
