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

from maudlin_core.src.lib.framework.maudlin import load_yaml_file, save_yaml_file, load_json_file, save_json_file


def initialize_training_run_directory(maudlin, config):
    # Define paths
    unit_dir = os.path.join(maudlin['data-directory'], 'trainings', maudlin['current-unit'])
    os.makedirs(unit_dir, exist_ok=True)

    # Counter file path
    counter_file = os.path.join(unit_dir, 'run_metadata.json')

    is_brand_new =True

    # Initialize metadata
    if os.path.exists(counter_file):
        is_brand_new = False
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

    # Copy config file to the current run directory
    prev_config_path = ''

    if config['use_existing_model']:
        prev_config_path = os.path.join(unit_dir, f"run_{prev_curr_run_id}/config.yaml")
    else:
        prev_config_path = os.path.join(maudlin['data-directory'], 'configs', maudlin['current-unit'] + ".config.yaml")

    prev_config_path = os.path.join(unit_dir, f"run_{prev_curr_run_id}/config.yaml") if not is_brand_new else os.path.join(maudlin['data-directory'], 'configs', maudlin['current-unit'] + ".config.yaml")
    shutil.copy(prev_config_path, data_dir + "/config.yaml")

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
    parser.add_argument(
        "--use-existing-model", "-m",
        action="store_true",
        help="Use the existing model if it exists in the training run directory."
    )
    args = parser.parse_args(cli_args) if cli_args is not None else parser.parse_args()

    # Load configuration
#    if not args.training_run_path:
 #       raise ValueError("training_run_path (likely the path to the most recent training run) is required")

    config = get_current_unit_config(args.training_run_path)
    config['mode'] = 'training'
    config['use_existing_model'] = args.use_existing_model

    if (config['use_existing_model']):
        print("---- using EXISTING model")
    else:
        print("---- training A NEW model")

    maudlin = load_maudlin_data()

    # Setup directories
    #config_path = args.training_run_path # or maudlin['data-directory'] + get_current_unit_properties(maudlin)['config-path']
    data_dir, run_id, parent_run_id = initialize_training_run_directory(maudlin, config)
    config['run_id'] = run_id
    config['parent_run_id'] = parent_run_id

    """Configure training callbacks"""
    metrics_to_track = config.get("metrics", ["mae"])
    alrconfig = config['training']['adaptive_learning_rate']

    callbacks = [
        AdaptiveLearningRate(
            metric_name=metrics_to_track[0],
            patience=alrconfig['patience'],
            factor=alrconfig['factor'],
            min_lr=alrconfig['min-lr'],
            reduction_grace_period=alrconfig['reduction_grace_period']
        ),
        TrackBestMetric(metric_names=metrics_to_track, log_dir=data_dir),
        TensorBoard(log_dir=data_dir + "/tensorboard", histogram_freq=1)
    ]

    # Initialize training manager
    training_manager = DataPreparationManager(config, data_dir)
    setup_signal_handler(training_manager)

    # Update epochs in the config if specified
    if hasattr(args, 'epochs') and args.epochs is not None:
        config['epochs'] = args.epochs
        save_yaml_file(config, data_dir + "/config.yaml")
        print(f"Updated epochs to {args.epochs} in {data_dir}/config.yaml")

    try:
        # Load and prepare data
        X_train, y_train, X_test, y_test, X_val, y_val = training_manager.load_and_prepare_data()
        
        # Setup and train model
        training_manager.setup_model(X_train.shape[-1], config)
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


# Iterate over diffs in File C
def apply_patch(diff_content, target_file):
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as temp_diff:
        temp_diff.writelines(diff_content)
        temp_diff_path = temp_diff.name

    subprocess.run(["patch", target_file, temp_diff_path])


def apply_config_changes(config_file, changes):
    """
    Applies a set of changes to the configuration file.
    """
    # Read File C and apply each diff to File A
    with open(file_c, 'r') as f_c:
        diffs = f_c.read().split('\n---\n')  # Split diffs by separator (adjust based on format)
        for diff_content in diffs:
            if diff_content.strip():
                apply_patch(diff_content, file_a)
                subprocess.run(["your_process", file_a])  # Replace "your_process" with actual command


    updated_config = always_merger.merge(original_config, changes['diff'])

    save_yaml_file(config, config_file)

def load_batch_changes(batch_file):
    # batch file is json, in the format {"comment": "some comment", "diff": ["diff line 1", "diff line 2", ...]}
    with open(batch_file, 'r') as f:
        return json.load(f)

def process_batch_config_changes():
    batch_file = os.path.join(DEFAULT_DATA_DIR, 'batch_config_changes.txt')
    trained_file = os.path.join(DEFAULT_DATA_DIR, 'batch_config_changes.txt.trained')

    if not os.path.exists(batch_file):
        print("No batch_config_changes.txt file found. Running regular training.")
        run_batch_training()
        return

    # Load batch changes
    batch_changes = load_batch_changes(batch_file)
    if not batch_changes:
        print("No changes to process in batch_config_changes.yaml.")
        return

    # Process changes one by one
    for change_set in batch_changes:
        print(f"Applying changes: {change_set['comment']}\n")

        # Determine the current unit config path
        config_file = os.path.join(DEFAULT_DATA_DIR, 'configs', f"{CURRENT_UNIT}.config.yaml")
        apply_patch(change_set['diff'], config_file)

        # Run training
        run_batch_training()

        # Move processed change to the trained file
        batch_changes.remove(change_set)
        save_json_file(batch_changes, batch_file)

        trained_changes = load_yaml_file(trained_file) or []
        trained_changes.append(change_set)
        save_json_file(trained_changes, trained_file)

if __name__ == "__main__":
    process_batch_config_changes()