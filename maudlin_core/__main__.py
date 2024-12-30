import os
import sys
import yaml
import subprocess
from pathlib import Path

DEFAULT_DATA_DIR = os.path.expanduser("~/src/_data/maudlin")
DEFAULT_CONFIG_FILE = "default.config.yaml"
DATA_YAML = os.path.join(DEFAULT_DATA_DIR, "maudlin.data.yaml")


class MaudlinCLI:
    def __init__(self):
        self.current_unit = self.get_current_unit()

    def get_current_unit(self):
        if os.path.exists(DATA_YAML):
            with open(DATA_YAML, 'r') as file:
                data = yaml.safe_load(file)
                return data.get('current-unit')
        return None

    def save_yaml(self, data):
        with open(DATA_YAML, 'w') as file:
            yaml.safe_dump(data, file)

    def initialize_maudlin(self):
        print("Initializing Maudlin directory structure...")
        dirs = [
            "configs",
            "models",
            "functions",
            "inputs",
            "predictions",
        ]

        for d in dirs:
            Path(f"{DEFAULT_DATA_DIR}/{d}").mkdir(parents=True, exist_ok=True)

        if not os.path.exists(DATA_YAML):
            with open(DATA_YAML, 'w') as f:
                yaml.dump({"units": [], "data-directory": DEFAULT_DATA_DIR}, f)

        print("Maudlin initialization complete.")

    def list_units(self):
        if os.path.exists(DATA_YAML):
            with open(DATA_YAML, 'r') as file:
                data = yaml.safe_load(file)
                print("Units:", list(data.get('units', {}).keys()))
        else:
            print("Maudlin data file not found. Initialize first using 'mdln init'.")

    def set_current_unit(self, unit_name):
        with open(DATA_YAML, 'r') as file:
            data = yaml.safe_load(file)
            if unit_name not in data['units']:
                print(f"Error: Unit '{unit_name}' does not exist.")
                return
            data['current-unit'] = unit_name
            self.save_yaml(data)
            print(f"Current unit set to '{unit_name}'.")

    def new_unit(self, unit_name, training_csv, prediction_csv):
        with open(DATA_YAML, 'r') as file:
            data = yaml.safe_load(file)
            if unit_name in data['units']:
                print(f"Error: Unit '{unit_name}' already exists.")
                return

        # Create config and function files
        unit_dir = os.path.join(DEFAULT_DATA_DIR, 'functions', unit_name)
        Path(unit_dir).mkdir(parents=True, exist_ok=True)

        # Config setup
        config_path = os.path.join(DEFAULT_DATA_DIR, 'configs', f"{unit_name}.config.yaml")
        with open(config_path, 'w') as config_file:
            yaml.dump({
                'training_file': training_csv,
                'prediction_file': prediction_csv
            }, config_file)

        # Update data yaml
        data['units'][unit_name] = {
            'config-path': f"configs/{unit_name}.config.yaml",
            'input-function': f"functions/{unit_name}/input.py",
            'target-function': f"functions/{unit_name}/target.py"
        }
        self.save_yaml(data)

        print(f"Unit '{unit_name}' created successfully.")

    def show_current_unit(self):
        if not self.current_unit:
            print("No current unit is set. Use 'mdln use <unit>' to set a unit.")
            return

        with open(DATA_YAML, 'r') as file:
            data = yaml.safe_load(file)
            unit_data = data['units'].get(self.current_unit, {})
            print(f"Current Unit: {self.current_unit}")
            for key, value in unit_data.items():
                print(f"  {key}: {value}")

    def run_training(self, epochs=None):
        config_path = os.path.join(DEFAULT_DATA_DIR, 'configs', f"{self.current_unit}.config.yaml")
        if epochs:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                config['epochs'] = epochs
            with open(config_path, 'w') as file:
                yaml.safe_dump(config, file)
        print(f"Training unit '{self.current_unit}'...")
        subprocess.run(["python3", "maudlin-core/training/batch.py", f"--config={config_path}"])

    def run_predictions(self):
        print(f"Running predictions for unit '{self.current_unit}'...")
        subprocess.run(["python3", "maudlin-core/predicting/predict.py"])

    def edit_current_unit(self):
        config_path = os.path.join(DEFAULT_DATA_DIR, 'configs', f"{self.current_unit}.config.yaml")
        function_path = os.path.join(DEFAULT_DATA_DIR, 'functions', self.current_unit)
        subprocess.run(["vim", config_path, function_path])

    def clean_output(self):
        model_path = os.path.join(DEFAULT_DATA_DIR, 'models', f"{self.current_unit}.h5")
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"Model for '{self.current_unit}' removed.")
        else:
            print(f"No model found for '{self.current_unit}'.")

    def visualize_history(self):
        history_file = os.path.join(DEFAULT_DATA_DIR, 'trainings', self.current_unit, 'history.yaml')
        if not os.path.exists(history_file):
            print(f"No history file found for unit '{self.current_unit}'.")
            return
        print(f"Visualizing history for unit '{self.current_unit}'...")
        subprocess.run(["python3", "maudlin-core/exploring/visualize_history.py", history_file])


def main():
    cli = MaudlinCLI()

    commands = {
        'init': cli.initialize_maudlin,
        'list': cli.list_units,
        'use': lambda: cli.set_current_unit(sys.argv[2]),
        'new': lambda: cli.new_unit(sys.argv[2], sys.argv[3], sys.argv[4]),
        'show': cli.show_current_unit,
        'train': lambda: cli.run_training(epochs=int(sys.argv[2]) if len(sys.argv) > 2 else None),
        'predict': cli.run_predictions,
        'edit': cli.edit_current_unit,
        'clean': cli.clean_output,
        'history': cli.visualize_history
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("Usage: mdln {init | list | new | use | show | edit | clean | predict | train | history}")
        sys.exit(1)

    command = sys.argv[1]
    commands[command]()


if __name__ == "__main__":
    main()

