import os
import yaml
import json
import sys

MAUDLIN_DATA_DIR = os.path.expanduser("~/src/_data/maudlin")
MAUDLIN_DATA_FILE = os.path.join(MAUDLIN_DATA_DIR, "maudlin.data.yaml")

def load_maudlin_data():
    """Load the maudlin.data.yaml file."""
    if not os.path.exists(MAUDLIN_DATA_FILE):
        raise FileNotFoundError(f"Maudlin data file not found at {MAUDLIN_DATA_FILE}.")
    
    with open(MAUDLIN_DATA_FILE, "r") as file:
        return yaml.safe_load(file)

def save_maudlin_data(data):
    """Save the given data to the maudlin.data.yaml file."""
    if not os.path.exists(MAUDLIN_DATA_DIR):
        raise FileNotFoundError(f"Data directory not found at {MAUDLIN_DATA_DIR}.")

    with open(MAUDLIN_DATA_FILE, "w") as file:
        yaml.safe_dump(data, file)


def get_current_unit_name(data):
    """Get the current unit from the loaded maudlin data."""
    current_unit = data.get("current-unit")
    if not current_unit:
        raise ValueError("No current unit is set in maudlin.data.yaml.")
    return current_unit

def get_current_unit_properties(data):
    current_unit_name = get_current_unit_name(data)
    return get_unit_properties(data, current_unit_name)

def get_unit_properties(data, unit_name):
    """Get the properties of the specified unit."""
    return data.get("units").get(unit_name)

def get_unit_function_path(data, function_path_key):
    rtn = data['data-directory'] + get_current_unit_properties(data)[function_path_key]

    # Validate the file path
    if not os.path.isfile(rtn):
        print(f"Error: File '{rtn}' does not exist.")
        sys.exit(1)

    return rtn

def write_keras_filename_for_current_unit(data_dir, model_file_name):
    """Update the keras filename for the current unit in maudlin.data.yaml."""
    # Load the maudlin data
    data = load_maudlin_data()

    # Get the current unit name
    current_unit_name = get_current_unit_name(data)
    updated = False

    # Find and update the current unit's keras-filename
    for unit in data.get("units", []):
        if unit == current_unit_name:
            data.get("units").get(current_unit_name)['keras-filename'] = f"/models/{model_file_name}"
            updated = True
            break

    if not updated:
        raise ValueError(f"Unit '{current_unit_name}' not found in maudlin.data.yaml.")

    # Save the updated data
    save_maudlin_data(data)

    print(f"Keras filename for '{current_unit_name}' updated to '/models/{model_file_name}'.")

    return data["data-directory"] + "/models/" + model_file_name

def load_json_file(path):
    """Utility to load JSON from a file."""
    with open(path, 'r') as f:
        return json.load(f)

def save_json_file(data, path):
    """Utility to write data to a JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_yaml_file(path):
    """Utility to load YAML from a file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml_file(data, path):
    """Utility to write data to a YAML file."""

    # Create a custom Dumper class for safe_dump
    class InlineListSafeDumper(yaml.SafeDumper):
        def increase_indent(self, flow=False, indentless=False):
            return super(InlineListSafeDumper, self).increase_indent(flow=True, indentless=indentless)

    # Dump the YAML using safe_dump with inline lists
    yaml_output = yaml.dump(data, Dumper=InlineListSafeDumper, default_flow_style=None)

    with open(path, 'w') as file:
        file.write(yaml_output)

def pretty_print_diff(diff):
    for line in diff:
        if line.startswith('---') or line.startswith('+++'):
            print(f'\033[1;34m{line}\033[0m', end='')  # Blue for file headers
        elif line.startswith('@@'):
            print(f'\033[1;33m{line}\033[0m', end='')  # Yellow for hunk headers
        elif line.startswith('+'):
            print(f'\033[1;32m{line}\033[0m', end='')  # Green for additions
        elif line.startswith('-'):
            print(f'\033[1;31m{line}\033[0m', end='')  # Red for deletions
        else:
            print(line, end='')  # Default color for context lines

def get_current_training_run_id(data):
    DEFAULT_DATA_DIR = os.path.expanduser("~/src/_data/maudlin")
    CURRENT_UNIT = get_current_unit_name(data)
    run_metadata_path = os.path.join(DEFAULT_DATA_DIR, 'trainings', CURRENT_UNIT, 'run_metadata.json')
    rtn = None
    if os.path.exists(run_metadata_path):
        with open(run_metadata_path, 'r') as f:
            run_metadata = yaml.safe_load(f)
            rtn = run_metadata.get('current_run_id', None)

    return rtn
