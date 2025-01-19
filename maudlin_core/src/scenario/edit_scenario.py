import os
import json
import tempfile
import shutil
import subprocess
import difflib

from pathlib import Path
from maudlin_core.src.lib.framework.maudlin import load_maudlin_data, pretty_print_diff
from maudlin_core.src.scenario.diff_to_sed_commands import diff_to_sed_commands

def main():
    # Load the Maudlin metadata
    maudlin = load_maudlin_data()

    # Paths
    config_path = os.path.join(
        maudlin['data-directory'],
        'configs',
        maudlin['current-unit'] + ".config.yaml"
    )
    unit_name = maudlin['current-unit']
    unit_dir = os.path.join(maudlin['data-directory'], 'trainings', unit_name)
    batch_config_path = os.path.join(unit_dir, 'batch_config_changes.txt')

    # Make sure we have a batch_config_changes file
    if not os.path.exists(batch_config_path):
        print(f"\nNo scenarios have been set up yet ({batch_config_path} is missing).")
        return

    # Load existing scenarios
    with open(batch_config_path, 'r') as f:
        try:
            scenarios = json.load(f)
            if not isinstance(scenarios, list):
                raise ValueError("batch_config_changes.txt does not contain a list.")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error reading {batch_config_path}: {e}")
            return

    if not scenarios:
        print("\nNo scenarios found in batch_config_changes.txt.\n")
        return

    # List scenarios
    print("\nExisting scenarios:\n")
    for i, scenario in enumerate(scenarios):
        comment = scenario.get("comment", "<no comment>")
        print(f"{i}. {comment}")
    print()

    # Let user pick a scenario
    try:
        choice = int(input("Enter the index of the scenario you want to edit: "))
        if not (0 <= choice < len(scenarios)):
            print("Invalid scenario index.")
            return
    except ValueError:
        print("Invalid input; please enter an integer.")
        return

    scenario = scenarios[choice]
    original_comment = scenario.get("comment", "")
    original_sed = scenario.get("sed_commands", [])
    original_optimize = scenario.get("optimize", False)

    # Check if the config file exists
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    # Step 1: Copy the current config into a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as temp_file:
        temp_config_path = temp_file.name
    shutil.copy(config_path, temp_config_path)

    # Step 2 - Create a sed command for each entry in original_sed and run it on the temp config
    for sed_command in original_sed:
        try:
            # Apply the sed command to the temp config file
            subprocess.run(
                ['sed', '-i', sed_command, temp_config_path],
                check=True,
                text=True
            )
            print(f"Applied sed command: {sed_command}")
        except subprocess.CalledProcessError as e:
            print(f"Error applying sed command: {sed_command}")
            print(e)

    # Step 3: Open the patched config in an editor for further changes
    editor = "vim"  # or "nano", "code", etc.
    subprocess.run([editor, temp_config_path])

    # Step 4: Compare the edited file (temp_config_path) with the *original* config
    with open(config_path, 'r') as f_original:
        original_config_lines = f_original.readlines()
    with open(temp_config_path, 'r') as f_edited:
        edited_config_lines = f_edited.readlines()

    new_diff = list(difflib.unified_diff(
        original_config_lines,
        edited_config_lines,
        fromfile=config_path,
        tofile=temp_config_path
    ))

    # Step 6: If there are changes, ask user for updated comment, optimize
    if new_diff:
        print("\nDiff found from the original config. Updating scenario...\n")
        pretty_print_diff(new_diff)
        print()

        print("Current comment:", original_comment)
        new_comment = input("Enter new comment (or press Enter to keep current): ").strip()
        if new_comment:
            scenario["comment"] = new_comment

        print(f"Current optimize setting: {original_optimize} (True/False)")
        opt_choice = input("Do you want to optimize before training? (y/n, Enter to keep current): ").strip().lower()
        if opt_choice.startswith('y'):
            scenario["optimize"] = True
        elif opt_choice.startswith('n'):
            scenario["optimize"] = False
        # if user presses Enter, keep the old setting

        # Update the diff in the scenario
        scenario["sed_commands"] = diff_to_sed_commands(new_diff)

        # Step 7: Write the updated array back to batch_config_changes.txt
        with open(batch_config_path, 'w') as f:
            json.dump(scenarios, f, indent=4)

        print("Scenario updated.\n")
    else:
        print("\nNo changes were made to the config (diff is empty). Scenario not updated.")

    # Clean up temp files
    os.unlink(temp_config_path)


if __name__ == "__main__":
    main()
