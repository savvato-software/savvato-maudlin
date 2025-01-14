import os
import shutil
import tempfile
import subprocess
from pathlib import Path
import difflib

from maudlin_core.src.lib.framework.maudlin import load_maudlin_data

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

        obj = {
            "comment": comment,
            "diff": diff
        }

        unit_name = maudlin['current-unit']
        unit_dir = os.path.join(maudlin['data-directory'], 'trainings', unit_name)

        if not os.path.exists(unit_dir + '/batch_config_changes.txt'):
            with open(unit_dir + '/batch_config_changes.txt', 'w') as f_c:
                f_c.writelines(str(obj))
                f_c.write("\n")
        else:
            with open(unit_dir + '/batch_config_changes.txt', 'a') as f_c:
                f_c.writelines(str(obj))
                f_c.write("\n")

        print("Scenario changes saved.")
        pretty_print_diff(diff)

    # remove the temp file
    os.unlink(file_b)


if __name__ == "__main__":
    main()
