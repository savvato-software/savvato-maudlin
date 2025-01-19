
import os
import yaml
import subprocess
from datetime import datetime


def update_history(config, data_dir):
    run_id = config['runtime']['run_id']
    parent_id = config['runtime']['parent_run_id']
    history_path = os.path.join(data_dir, "history.yaml")
    config_path = os.path.join(data_dir, f"run_{str(run_id)}", "config.yaml")
    parent_config_path = os.path.join(data_dir, f"run_{str(parent_id)}", "config.yaml") if parent_id else None

    # Load existing history
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = yaml.safe_load(f)
    else:
        history = {"history": []}


    # Generate config diff (use Unix diff command)
    config_diff = ""
    if parent_config_path:
        result = subprocess.run(
            ["diff", "-u", parent_config_path, config_path], capture_output=True, text=True
        )
        config_diff = result.stdout.strip()

    # Create new entry
    new_entry = {
        "id": run_id,
        "parent": parent_id if parent_id > 0 else None,
        "timestamp": datetime.now().isoformat(),
        "config_diff": config_diff,
        "message": "",  # Manual message via 'mdln msg'
        "children": [],
    }

    # Append to history and update parent-child relationships
    history['history'].append(new_entry)
    for entry in history['history']:
        if entry['id'] == parent_id:
            entry['children'].append(run_id)

    # Save history
    with open(history_path, "w") as f:
        yaml.safe_dump(history, f)

