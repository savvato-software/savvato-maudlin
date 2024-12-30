import os
import sys
import yaml
import argparse
import subprocess
from pathlib import Path

DEFAULT_DATA_DIR = os.path.expanduser("~/src/_data/maudlin")
DEFAULT_CONFIG_FILE = "default.config.yaml"
DATA_YAML = os.path.join(DEFAULT_DATA_DIR, "maudlin.data.yaml")

# Set CURRENT_UNIT from YAML
with open(DATA_YAML, 'r') as f:
    data_yaml_content = yaml.safe_load(f)
CURRENT_UNIT = data_yaml_content.get('current-unit', 'default')

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


    def run_training(self, epochs=None, training_run_id=None):
        # Retrieve config path for the current unit
        with open(DATA_YAML, 'r') as f:
            config = yaml.safe_load(f)

        config_path = config.get('units', {}).get(CURRENT_UNIT, {}).get('config-path').lstrip("/")
        if not config_path:
            print(f"Error: Config path not found for the current unit '{CURRENT_UNIT}'.")
            sys.exit(1)

        full_config_path = os.path.join(DEFAULT_DATA_DIR, config_path)

        # Ensure the config file exists before proceeding
        if not os.path.isfile(full_config_path):
            print(f"Error: Config file '{full_config_path}' does not exist.")
            sys.exit(1)

        with open(full_config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Update epochs in the config if specified
        if epochs is not None:
            config_data['epochs'] = epochs
            with open(full_config_path, 'w') as f:
                yaml.safe_dump(config_data, f)
            print(f"Updated epochs to {epochs} in {full_config_path}")

        # Retrieve USE_ONLINE_LEARNING_MODE
        use_online_learning_mode = config_data.get('use_online_learning', False)

        # Handle training run ID
        if training_run_id:
            training_run_path = os.path.join(DEFAULT_DATA_DIR, 'trainings', CURRENT_UNIT, f'run_{training_run_id}')

            if not os.path.isdir(training_run_path):
                print(f"Error: Training run directory '{training_run_path}' does not exist.")
                sys.exit(1)

            if use_online_learning_mode:
                subprocess.run(["python3", "-m", "maudlin_core.training.online_learn", training_run_path], check=True)
            else:
                subprocess.run(["python3", "-m", "maudlin_core.training.batch", training_run_path], check=True)
        else:
            # Execute training without parameters
            if use_online_learning_mode:
                subprocess.run(["python3", "-m", "maudlin_core.training.online_learn"], check=True)
            else:
                subprocess.run(["python3", "-m", "maudlin_core.training.batch"], check=True)

    def run_predictions(self):
        print(f"Running predictions for unit '{self.current_unit}'...")
        subprocess.run(["python3", "-m", "maudlin-core.predicting.predict"])

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


    def visualize_history(self, interactive=False, tree=False, list_view=False):
        history_file = os.path.join(DEFAULT_DATA_DIR, 'trainings', self.current_unit, 'history.yaml')
        if not os.path.exists(history_file):
            print(f"No history file found for unit '{self.current_unit}'.")
            return

        print(f"Visualizing history for unit '{self.current_unit}'...")

        # Build subprocess arguments
        cmd = ["python3", "-m", "maudlin_core.exploring.visualize_history", history_file]

        # Add flags
        if interactive:
            cmd.append('--interactive')
        elif tree:
            cmd.append('--tree')
        elif list_view:
            cmd.append('--list')

        # Run subprocess
        subprocess.run(cmd)



def main():
    cli = MaudlinCLI()

    parser = argparse.ArgumentParser(
        description='Maudlin CLI',
        usage='mdln {init | list | new <unit-name> <training_csv> <prediction_csv> | use <unit-name> | show | edit | clean | predict | train [-e EPOCHS] | history}'
    )
    subparsers = parser.add_subparsers(dest='command')

    # Init Command
    subparsers.add_parser('init')

    # List Command
    subparsers.add_parser('list')

    # Use Command
    use_parser = subparsers.add_parser('use')
    use_parser.add_argument('unit_name', help='Name of the unit to use')

    # New Command
    new_parser = subparsers.add_parser('new')
    new_parser.add_argument('unit_name', help='Name of the new unit')
    new_parser.add_argument('training_csv', help='Path to training CSV file')
    new_parser.add_argument('prediction_csv', help='Path to prediction CSV file')

    # Show Command
    subparsers.add_parser('show')

    # Train Command
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('-e', '--epochs', type=int, default=None, help='Number of epochs')
    train_parser.add_argument('-r', '--run-id', type=str, default=None, help='Training run ID')

    # Predict Command
    subparsers.add_parser('predict')

    # Edit Command
    subparsers.add_parser('edit')

    # Clean Command
    subparsers.add_parser('clean')

    # History Command
    history_parser = subparsers.add_parser('history')
    history_parser.add_argument('-i', '--interactive', action='store_true', help='Scroll through training runs')
    history_parser.add_argument('-t', '--tree', action='store_true', help='Hierarchical view of training runs')
    history_parser.add_argument('-l', '--list', action='store_true', help='List training runs')


    # Parse Arguments
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    # Command Mapping
    commands = {
        'init': cli.initialize_maudlin,
        'list': cli.list_units,
        'use': lambda: cli.set_current_unit(args.unit_name),
        'new': lambda: cli.new_unit(args.unit_name, args.training_csv, args.prediction_csv),
        'show': cli.show_current_unit,
        'train': lambda: cli.run_training(epochs=args.epochs),
        'predict': cli.run_predictions,
        'edit': cli.edit_current_unit,
        'clean': cli.clean_output,
        'history': lambda: cli.visualize_history(
            interactive=args.interactive,
            tree=args.tree,
            list_view=args.list
            )
    } 

    # Execute Command
    if args.command:
        commands[args.command]()

if __name__ == "__main__":
    main()

