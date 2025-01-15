import os
import json
import subprocess

from maudlin_core.src.lib.framework.maudlin import load_maudlin_data, pretty_print_diff


def main():
    maudlin = load_maudlin_data()

    unit_name = maudlin['current-unit']
    unit_dir = os.path.join(maudlin['data-directory'], 'trainings', unit_name)

    if os.path.exists(unit_dir + '/batch_config_changes.txt'):
        # JSON array of changes, each change is an object with keys 'comment', 'diff', and 'optimize'. read and display
        with open(unit_dir + '/batch_config_changes.txt', 'r') as f:
            changes = json.load(f)

        print("\nCurrent batch_config_changes.txt contents:\n")
        for i, change in enumerate(changes):
            print(f"Change {i + 1}: {change['comment']}\n")
            print(f"Optimize before training: {'Yes' if change['optimize'] else 'No'}\n")
            pretty_print_diff(change['diff'])
    else:
        print("\nNo scenarios have been set up yet.\n")

if __name__ == "__main__":
    main()