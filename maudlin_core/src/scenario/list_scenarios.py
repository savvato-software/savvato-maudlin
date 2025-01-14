import os
import subprocess

from maudlin_core.src.lib.framework.maudlin import load_maudlin_data


def main():
    unit_name = maudlin['current-unit']
    unit_dir = os.path.join(maudlin['data-directory'], 'training', unit_name)

    if os.path.exists(unit_dir + '/batch_config_changes.txt'):
        subprocess.run(["less", unit_dir + '/batch_config_changes.txt'])
    else:
        print("\nNo scenarios have been set up yet.\n")

if __name__ == "__main__":
    main()