import yaml
import sys
from maudlin_unit_config import get_current_unit_config

# Load and Save History
def load_history(config):
    history_path = f"{config['data-directory']}/trainings/{config['unit']}/history.yaml"
    with open(history_path, "r") as f:
        return yaml.safe_load(f), history_path

def save_history(history_path, history):
    with open(history_path, "w") as f:
        yaml.safe_dump(history, f)

# Add Message to Current Run
def add_message(config, message):
    # Load history and config
    history, history_path = load_history(config)
    current_run_id = config['current-run']  # Assume config tracks 'current-run'

    # Find the current run
    current_run = next(run for run in history['history'] if run['id'] == current_run_id)
    current_run['message'] = message  # Update message

    # Save updated history
    save_history(history_path, history)
    print(f"Added message to {current_run_id}: {message}")

# Main
def main():
    # Get config and message
    config = get_current_unit_config()
    message = sys.argv[1]

    # Update history
    add_message(config, message)

if __name__ == "__main__":
    main()

