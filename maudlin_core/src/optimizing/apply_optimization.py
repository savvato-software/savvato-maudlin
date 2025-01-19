import os
import sys
import json
from maudlin_core.src.lib.framework.maudlin_unit_config import get_current_unit_config
from maudlin_core.src.lib.framework.maudlin import load_maudlin_data
from maudlin_core.src.lib.framework.maudlin import load_yaml_file, save_yaml_file


def locate_best_trials_file(maudlin, config):
    """Determine the path to the best_trials.yaml file using run_metadata.json."""
    unit_name = maudlin['current-unit']
    unit_dir = os.path.join(maudlin['data-directory'], 'optimizations', unit_name)

    # Load run metadata
    metadata_path = os.path.join(unit_dir, 'run_metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Run metadata file not found at {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Identify the latest run directory
    current_run_id = metadata.get('current_run_id')
    if current_run_id is None:
        raise ValueError("current_run_id not found in run_metadata.json")

    best_trials_file = os.path.join(unit_dir, f"run_{current_run_id}", "best_trials.yaml")
    if not os.path.exists(best_trials_file):
        raise FileNotFoundError(f"best_trials.yaml not found at {best_trials_file}")

    return best_trials_file


def apply_optimization(best_trial_index=1, output_file=None):
    # 2. Load Maudlin configuration
    maudlin = load_maudlin_data()

    # 3. Locate the best_trials.yaml file
    config = get_current_unit_config()
    best_trials_file = locate_best_trials_file(maudlin, config)

    # 4. Load best_trials
    best_trials = load_yaml_file(best_trials_file)

    # 5. Validate the index and retrieve the chosen trial
    if best_trial_index < 1 or best_trial_index > len(best_trials):
        print(f"Invalid best_trial_index={best_trial_index}. Must be between 1 and {len(best_trials)}.")
        sys.exit(1)

    chosen_trial = best_trials[best_trial_index-1]

    # 6. Load the current config
    config_file = os.path.join(
        maudlin['data-directory'], 'configs', maudlin['current-unit'] + ".config.yaml"
    )
    config = load_yaml_file(config_file)

    # 7. Update config with trial parameters
    config['model_architecture'] = chosen_trial['model_architecture']

    params = chosen_trial.get('params', {})
    config['batch_size'] = params.get('batch_size', config.get('batch_size', 32))
    config['learning_rate'] = params.get('lr', config.get('learning_rate', 1e-3))
    config['optimizer'] = params.get('optimizer', config.get('optimizer', 'adam'))

    # Identify used feature indices
    used_feature_indices = [
        int(key.split('_')[1]) for key, value in params.items() if key.startswith('feature_') and value == 1
    ]

    # Ensure pre_training section exists
    if 'pre_training' not in config:
        config['pre_training'] = {}
    config['pre_training']['optimized_feature_indices'] = used_feature_indices

    # 8. Save the updated config
    if not output_file:
        output_file = config_file  # Default to overwriting the original config

    save_yaml_file(config, output_file)
    print(f"Updated config saved to: {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python apply_optimization.py <best_trial_index> [output_file]")
        sys.exit(1)

    # 1. Capture command-line inputs
    best_trial_index = int(sys.argv[1])  # Convert 1-based index to 0-based

    # Optional: allow specifying a different output path for the updated config
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    apply_optimization(best_trial_index, output_file)

if __name__ == "__main__":
    main()
