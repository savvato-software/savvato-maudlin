import os
import signal
import sys
import json
import numpy as np
import shutil
import argparse
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

class TrainingManager:
    def __init__(self, config, data_dir):
        self.config = config
        self.data_dir = data_dir
        self.model = None
        self.model_file = None
        
    def load_and_prepare_data(self):
        """Handle data loading and preprocessing"""
        X, y, feature_count, columns = load_for_training(self.config, self.data_dir)
        
        # Split data into training, validation, and testing
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
        
        # Execute pre-training stage
        X_train, y_train, X_test, y_test, X_val, y_val = execute_pretraining_stage(
            self.config, self.data_dir, X_train, y_train, X_test, y_test, X_val, y_val, columns
        )
        
        return X_train, y_train, X_test, y_test, X_val, y_val

    def setup_model(self, input_shape):
        """Create and configure the model"""
        self.model = create_model(self.config, self.data_dir, input_shape)
        self.model_file = generate_model_file_name(self.config, self.data_dir)
        return self.model

    def setup_callbacks(self):
        """Configure training callbacks"""
        metrics_to_track = self.config.get("metrics", ["mae"])
        alrconfig = self.config['training']['adaptive_learning_rate']
        
        return [
            AdaptiveLearningRate(
                metric_name=metrics_to_track[0], 
                patience=alrconfig['patience'], 
                factor=alrconfig['factor'], 
                min_lr=alrconfig['min-lr']
            ),
            TrackBestMetric(metric_names=metrics_to_track, log_dir=self.data_dir),
            TensorBoard(log_dir=self.data_dir + "/tensorboard", histogram_freq=1)
        ]

    def get_class_weights(self):
        """Extract class weights from config"""
        classes = self.config.get('classes', [])
        return {cls['label']: cls['weight'] for cls in classes}

    def train_model(self, X_train, y_train, X_val, y_val):
        """Execute model training"""
        callbacks = self.setup_callbacks()
        class_weights = self.get_class_weights()
        
        history = self.model.fit(
            X_train, y_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weights
        )
        
        return history

    def save_model(self):
        """Save the trained model"""
        if self.model and self.model_file:
            self.model.save(self.model_file)
            return True
        return False

    def copy_config_file(self, source_config_path):
        """Copy the config file to the training directory"""
        dest_path = os.path.join(self.data_dir, "config.yaml")
        shutil.copy(source_config_path, dest_path)

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
        "training_run_path",
        nargs="?",
        default=None,
        help="Path to the training run directory (optional).",
    )
    args = parser.parse_args(cli_args) if cli_args is not None else parser.parse_args()

    # Load configuration
    config = get_current_unit_config(args.training_run_path)
    config['mode'] = 'training'
    maudlin = load_maudlin_data()

    # Setup directories
    config_path = args.training_run_path or maudlin['data-directory'] + get_current_unit_properties(maudlin)['config-path']
    data_dir, run_id, parent_run_id = initialize_training_run_directory(maudlin)
    config['run_id'] = run_id
    config['parent_run_id'] = parent_run_id

    # Initialize training manager
    training_manager = TrainingManager(config, data_dir)
    setup_signal_handler(training_manager)

    # Copy config file
    training_manager.copy_config_file(config_path)

    try:
        # Load and prepare data
        X_train, y_train, X_test, y_test, X_val, y_val = training_manager.load_and_prepare_data()
        
        # Setup and train model
        training_manager.setup_model(X_train.shape[-1])
        training_manager.train_model(X_train, y_train, X_val, y_val)
        
        # Save model and execute post-training
        training_manager.save_model()
        execute_posttraining_stage(config, data_dir, training_manager.model, X_train, y_train, X_test, y_test)

    except KeyboardInterrupt:
        training_manager.save_model()
        print(f"Training interrupted. Model saved to {training_manager.model_file}. Exiting.")

    print("run_batch_training completed successfully!")

if __name__ == "__main__":
    run_batch_training()
