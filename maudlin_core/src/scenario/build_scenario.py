import os
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
import difflib

from maudlin_core.src.lib.framework.maudlin import load_maudlin_data, pretty_print_diff


def main():
    # copy the unit config to temporary file
    maudlin = load_maudlin_data()

    config_path = os.path.join(maudlin['data-directory'], 'configs', maudlin['current-unit'] + ".config.yaml")

    # Create a temporary file B
    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt") as temp_file:
        file_b = temp_file.name
        shutil.copyfile(config_path, file_b)

    # open temp file in an editor
    editor = "vim"  # or "nano", "code", etc.
    subprocess.run([editor, file_b])

    # wait for the editor to close
    # if the temp file was modified, take the diff and update batch_config_changes.txt in the unit root directory
    # Compare the config file and File B
    with open(config_path, 'r') as f_a, open(file_b, 'r') as f_b:
        diff = list(difflib.unified_diff(
            f_a.readlines(),
            f_b.readlines(),
            fromfile=config_path,
            tofile=file_b
        ))

    # If there are differences, append them to File C
    if diff:
        # ask the user to enter a comment
        comment = input("Enter a comment for the changes: ")

        # ask yes no whether to optimize before training
        optimize = input("Do you want to optimize before training? (y/n): ")

        obj = {
            "comment": comment,
            "diff": diff,
            "optimize": optimize.lower().startswith('y')
        }

        unit_name = maudlin['current-unit']
        unit_dir = os.path.join(maudlin['data-directory'], 'trainings', unit_name)

        batch_config_path = os.path.join(unit_dir, 'batch_config_changes.txt')

        # Ensure the directory exists
        os.makedirs(unit_dir, exist_ok=True)

        # Initialize the data array
        data = []

        # Read the existing file if it exists
        if os.path.exists(batch_config_path):
            with open(batch_config_path, 'r') as f_c:
                try:
                    data = json.load(f_c)
                    if not isinstance(data, list):  # Ensure the file contains a list
                        data = []
                except json.JSONDecodeError:
                    # Handle corrupt or non-JSON file content
                    data = []

        # Append the new object to the array
        data.append(obj)

        # Write the updated array back to the file
        with open(batch_config_path, 'w') as f_c:
            json.dump(data, f_c, indent=4)

        print("Scenario changes saved.")
        pretty_print_diff(diff)

    # remove the temp file
    os.unlink(file_b)


if __name__ == "__main__":
    main()
