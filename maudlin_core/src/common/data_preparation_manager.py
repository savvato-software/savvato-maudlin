from maudlin_core.src.lib.framework.stage_functions.pre_training_function import execute_pretraining_stage
from maudlin_core.src.model.model import create_model, generate_model_file_name
from maudlin_core.src.lib.data_loading.training import load_for_training
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model

import glob
import os
import shutil


class DataPreparationManager:
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

    def setup_model(self, input_shape, config=None):
        if config and config['runtime'] and config['runtime']['use_existing_model']:
            if self.data_dir:
                parent_dir = os.path.dirname(self.data_dir) + "/run_" + str(config['runtime']['parent_run_id'])
                val = glob.glob(os.path.join(parent_dir, "model_*.keras"))
                if val:
                    self.model_file = val[0]
                    self.model = load_model(self.model_file)

                    print()
                    print(" USING EXISTING MODEL: ", self.model_file)
                    print()

                    self.model_file = generate_model_file_name(self.config, self.data_dir)
                else:
                    raise ValueError("Could not find the parent model file.")
            else:
                raise ValueError("Data directory is required to load an existing model.")
        else:
            """Create and configure the model"""
            self.model = create_model(self.config, self.data_dir, input_shape)

            if self.data_dir:
                self.model_file = generate_model_file_name(self.config, self.data_dir)

        return self.model

    def get_class_weights(self):
        """Extract class weights from config"""
        classes = self.config.get('classes', [])
        return {cls['label']: cls['weight'] for cls in classes}

    def train_model(self, callbacks, X_train, y_train, X_val, y_val):
        """Execute model training"""
        #callbacks = self.setup_callbacks()
        class_weights = self.get_class_weights()

        history = self.model.fit(
            X_train, y_train,
            shuffle=self.config['shuffle'],
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weights
        )

        return history, self.model

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

        return source_config_path, dest_path
