import os
import sys
import yaml
import argparse
import subprocess
from pathlib import Path
from maudlin_core.src.lib.framework.maudlin import save_yaml_file, get_current_unit_name

DEFAULT_DATA_DIR = os.path.expanduser("~/src/_data/maudlin")
DEFAULT_CONFIG_FILE = "default.config.yaml"
DATA_YAML = os.path.join(DEFAULT_DATA_DIR, "maudlin.data.yaml")

# Set CURRENT_UNIT from YAML
with open(DATA_YAML, 'r') as f:
    data_yaml_content = yaml.safe_load(f)
CURRENT_UNIT = data_yaml_content.get('current-unit', 'default')

class MaudlinCLI:
    def __init__(self):
        pass

    def initialize_maudlin(self):
        print("Initializing Maudlin directory structure...")
        dirs = [
            "configs",
            "models",
            "functions",
            "inputs",
            "predictions",
            "trainings",
            "scenarios",
            "optimizations"
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
            save_yaml_file(data)
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
        save_yaml_file(data)

        print(f"Unit '{unit_name}' created successfully.")

    def show_current_unit(self):
        if not CURRENT_UNIT:
            print("No current unit is set. Use 'mdln use <unit>' to set a unit.")
            return

        with open(DATA_YAML, 'r') as file:
            data = yaml.safe_load(file)
            unit_data = data['units'].get(CURRENT_UNIT, {})
            print(f"Current Unit: {CURRENT_UNIT}")
            for key, value in unit_data.items():
                print(f"  {key}: {value}")


    def run_training(self, epochs=None, training_run_id=None, use_existing_model=False):
        # Retrieve config path for the current unit
        with open(DATA_YAML, 'r') as f:
            config = yaml.safe_load(f)

        config_path = config.get('units', {}).get(CURRENT_UNIT, {}).get('config-path', '').lstrip("/")
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
            save_yaml_file(config_data, full_config_path)
            print(f"Updated epochs to {epochs} in {full_config_path}")

        # Retrieve USE_ONLINE_LEARNING_MODE
        use_online_learning_mode = config_data.get('use_online_learning', False)

        # Determine training_run_id
        if not training_run_id:
            run_metadata_path = os.path.join(DEFAULT_DATA_DIR, 'trainings', CURRENT_UNIT, 'run_metadata.json')
            if os.path.exists(run_metadata_path):
                with open(run_metadata_path, 'r') as f:
                    run_metadata = yaml.safe_load(f)
                    training_run_id = run_metadata.get('current_run_id', None)

        training_run_path = ''
        if training_run_id:
            training_run_path = os.path.join(DEFAULT_DATA_DIR, 'trainings', CURRENT_UNIT, f'run_{training_run_id}')
            if not os.path.isdir(training_run_path):
                print(f"Error: Training run directory '{training_run_path}' does not exist.")
                sys.exit(1)

        # Build the command dynamically
        base_command = ["python3", "-m"]

        if use_online_learning_mode:
            base_command.append("maudlin_core.src.training.online_learn")
        else:
            base_command.append("maudlin_core.src.training.batch")

        # Add optional parameters
        if epochs is not None:
            base_command.extend(["-e", str(epochs)])

        if use_existing_model:
            base_command.append("--use-existing-model")
            base_command.append(training_run_path)

        # Execute the command
        print()
        print(f"*********** {base_command} ***********")
        print()
        subprocess.run(base_command, check=True)


    def run_predictions(self):
        print(f"Running predictions for unit '{CURRENT_UNIT}'...")
        subprocess.run(["python3", "-m", "maudlin_core.src.predicting.predict"])

    def list_files(self, base_dir, sub_path, pattern):
        path = os.path.join(base_dir, sub_path)
        import fnmatch
        rtn = [os.path.join(path, f) for f in os.listdir(path) if fnmatch.fnmatch(f, pattern)]

        return rtn
        
    def edit_current_unit(self):
        # Load YAML config
        with open(DATA_YAML, 'r') as f:
            config = yaml.safe_load(f)

        # Retrieve paths
        config_path = config.get('units', {}).get(CURRENT_UNIT, {}).get('config-path').lstrip("/")
        target_function_slug = config['units'][CURRENT_UNIT]['target-function'].lstrip("/")

        if not config_path or not target_function_slug:
            print(f"Error: Config or target-function paths not found for the current unit '{CURRENT_UNIT}'.")
            sys.exit(1)

        full_config_path = os.path.join(DEFAULT_DATA_DIR, config_path)
        full_target_function_path = os.path.join(DEFAULT_DATA_DIR, target_function_slug)

        # Verify files exist
        if not os.path.isfile(full_config_path):
            print(f"Error: Config file '{full_config_path}' does not exist.")
            sys.exit(1)

        if not os.path.isfile(full_target_function_path):
            print(f"Error: Target function file '{full_target_function_path}' does not exist.")
            sys.exit(1)

        # Open files in editor
        print(f"Opening config and function files for unit '{CURRENT_UNIT}' in lvim...")
        subprocess.run(["lvim", full_config_path, *self.list_files(DEFAULT_DATA_DIR, "functions", f"{CURRENT_UNIT}*.py"), "-p"])


    def clean_output(self):
        model_path = os.path.join(DEFAULT_DATA_DIR, 'models', f"{CURRENT_UNIT}.h5")
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"Model for '{CURRENT_UNIT}' removed.")
        else:
            print(f"No model found for '{CURRENT_UNIT}'.")


    def visualize_history(self, interactive=False, tree=False, list_view=False):
        history_file = os.path.join(DEFAULT_DATA_DIR, 'trainings', CURRENT_UNIT, 'history.yaml')
        if not os.path.exists(history_file):
            print(f"No history file found for unit '{CURRENT_UNIT}'.")
            return

        print(f"Visualizing history for unit '{CURRENT_UNIT}'...")

        # Build subprocess arguments
        cmd = ["python3", "-m", "maudlin_core.src.exploring.visualize_history", history_file]

        # Add flags
        if interactive:
            cmd.append('--interactive')
        elif tree:
            cmd.append('--tree')
        elif list_view:
            cmd.append('--list')

        # Run subprocess
        subprocess.run(cmd)

    def run_optimization(self, trials=100, use_existing_model=False, use_last_best_trials=False):
        if use_existing_model and use_last_best_trials:
            print("Error: Cannot use both --use-existing-model and --use-last-best-trials flags together.")
            sys.exit(1)

        base_command = ["python3", "-m", "maudlin_core.src.optimizing.optimize", "--trials", str(trials)]
        prev_run_id = None

        # Determine training_run_id
        run_metadata_path = os.path.join(DEFAULT_DATA_DIR, 'optimizations', CURRENT_UNIT, 'run_metadata.json')
        if os.path.exists(run_metadata_path):
            with open(run_metadata_path, 'r') as f:
                run_metadata = yaml.safe_load(f)
                prev_run_id = run_metadata.get('current_run_id', None)
        else:
            print(f"*********** ERROR: Could not find the run_metadata.json file. {run_metadata_path} ***********")

        prev_run_path = ''
        if prev_run_id:
            prev_run_path = os.path.join(DEFAULT_DATA_DIR, 'optimizations', CURRENT_UNIT, f'run_{prev_run_id}')
            if not os.path.isdir(prev_run_path):
                print(f"Error: Optimization run directory '{prev_run_path}' does not exist.")
                sys.exit(1)

        if use_existing_model:
            base_command.append("--use-existing-model")
            base_command.append(prev_run_path)
        elif use_last_best_trials:
            base_command.append("--use-last-best-trials")
            base_command.append(prev_run_path)


        # Execute the command
        print()
        print(f"*********** {base_command} ***********")
        print()
        subprocess.run(base_command, check=True)


    def apply_optimization(self, trial_index=1, output_file=None):
        if not output_file:
            cmd = ["python3", "-m", "maudlin_core.src.optimizing.apply_optimization", str(trial_index)]
        else:
            cmd = ["python3", "-m", "maudlin_core.src.optimizing.apply_optimization", str(trial_index), output_file]

        subprocess.run(cmd)

    def build_training_scenario(self):
        cmd = ["python3", "-m", "maudlin_core.src.scenario.build_scenario"]
        subprocess.run(cmd)

    def list_training_scenarios(self):
        cmd = ["python3", "-m", "maudlin_core.src.scenario.list_scenarios"]
        subprocess.run(cmd)

    def edit_training_scenario(self):
        cmd = ["python3", "-m", "maudlin_core.src.scenario.edit_scenario"]
        subprocess.run(cmd)

def main():
    cli = MaudlinCLI()

    parser = argparse.ArgumentParser(
        description='Maudlin CLI',
        usage='mdln {init | list | new <unit-name> <training_csv> <prediction_csv> | use <unit-name> | show | edit | clean | predict | train [-e EPOCHS] | history | optimize | apply-opt | build-training-scenario | list-training-scenarios}'
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
    train_parser.add_argument('-m', '--use-existing-model', action='store_true', help='Use existing model')

    # Optimize Command
    optimize_parser = subparsers.add_parser('optimize')
    optimize_parser.add_argument('-t', '--trials', type=int, default=100, help='Number of trials')
    optimize_parser.add_argument('-m', '--use-existing-model', action='store_true', help='Continue trials with the previous model')
    optimize_parser.add_argument('-lbt', '--use-last-best-trials', action='store_true', help='New trials constrained by specs from the last best trials of the previous model')

    # Apply Optimization Command
    aopt_parser = subparsers.add_parser('apply-opt')
    aopt_parser.add_argument('trial_index', type=int, help='Index of the trial to apply')
    aopt_parser.add_argument('output_file', nargs='?', default=None, help='Output file for the updated config')

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

    # Build Training Scenario Command
    subparsers.add_parser('build-training-scenario')

    # List Training Scenarios Command
    subparsers.add_parser('list-training-scenarios')

    # Edit Training Scenario Command
    subparsers.add_parser('edit-training-scenario')


    # Parse Arguments
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    # Command Mapping
    commands = {
        'init': cli.initialize_maudlin,
        'list': cli.list_units,
        'use': lambda: cli.set_current_unit(args.unit_name),
        'new': lambda: cli.new_unit(args.unit_name, args.training_csv, args.prediction_csv),
        'show': cli.show_current_unit,
        'train': lambda: cli.run_training(epochs=args.epochs, training_run_id=args.run_id, use_existing_model=args.use_existing_model),
        'predict': cli.run_predictions,
        'edit': cli.edit_current_unit,
        'clean': cli.clean_output,
        'history': lambda: cli.visualize_history(
            interactive=args.interactive,
            tree=args.tree,
            list_view=args.list
            ),
        'optimize': lambda: cli.run_optimization(trials=args.trials, use_existing_model=args.use_existing_model, use_last_best_trials=args.use_last_best_trials),
        'apply-opt': lambda: cli.apply_optimization(trial_index=args.trial_index, output_file=args.output_file),
        'build-training-scenario': lambda: cli.build_training_scenario(),
        'list-training-scenarios': lambda: cli.list_training_scenarios(),
        'edit-training-scenario': lambda: cli.edit_training_scenario()
    } 

    # Execute Command
    if args.command:
        commands[args.command]()

if __name__ == "__main__":
    main()

