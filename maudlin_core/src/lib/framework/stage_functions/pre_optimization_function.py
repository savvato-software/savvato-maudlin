import os
import yaml

from maudlin_core.src.lib.framework.maudlin import save_yaml_file

def execute_preoptimization_stage(config, unit_dir):
    rconfig = config.get('runtime')
    if 'use_last_best_trials' in rconfig and rconfig['use_last_best_trials']:
        set_optimization_config_by_previous_best_trials(config, unit_dir)

    # write the config file
    save_yaml_file(config, os.path.join(unit_dir, f"run_{rconfig['run_id']}", "config.yaml"))

def set_optimization_config_by_previous_best_trials(config, unit_dir):
    """When using an existing model, set the optimization configuration based on the previous best trials."""
    # Load the best trials
    rconfig = config.get('runtime')
    best_trials_file = os.path.join(unit_dir, f"run_{rconfig['parent_run_id']}", 'best_trials.yaml')

    if not os.path.exists(best_trials_file):
        raise FileNotFoundError(f"use_best_last_trials specified, but no best_trials.yaml found at {best_trials_file}")

    with open(best_trials_file, 'r') as f:
        best_trials = yaml.safe_load(f)

    # Initialize aggregators
    activations = set()
    optimizers = set()
    batch_sizes = []
    dropouts = []
    initial_units = []
    learning_rates = []
    n_layers = []

    # Flags for boolean settings
    has_batch_norm = False
    has_diminishing_units = False
    has_feature_masking = False

    # Process each trial
    for trial in best_trials:
        params = trial["params"]
        architecture = trial["model_architecture"]

        # Update activation and optimizer
        activations.add(params["activation"])
        optimizers.add(params["optimizer"])

        # Aggregate numeric parameters
        batch_sizes.append(params["batch_size"])
        dropouts.append(params["dropout"])
        initial_units.append(params["initial_layer_units"])
        learning_rates.append(params["lr"])
        n_layers.append(params["n_layers"])

        # Check for batch normalization layers
        if any(layer.get("layer_type") == "BatchNormalization" for layer in architecture):
            has_batch_norm = True

        # Check for diminishing units
        layer_units = [
            layer.get("units") for layer in architecture if "units" in layer
        ]
        if layer_units and layer_units != [layer_units[0]] * len(layer_units):
            has_diminishing_units = True

        # Check for feature masking
        for key, value in params.items():
            if key.startswith("feature_") and value == 0:
                has_feature_masking = True

    # Set activation and optimizer lists
    config["optimization"]["activation"] = sorted(activations)
    config["optimization"]["optimizer"] = sorted(optimizers)

    # Calculate ranges for numerical parameters
    def calculate_range(values):
        return {"min": min(values), "max": max(values)}

    config["optimization"]["batch_size"] = calculate_range(batch_sizes)
    config["optimization"]["dropout"] = calculate_range(dropouts)
    config["optimization"]["initial_layer_units"] = calculate_range(initial_units)
    config["optimization"]["learning_rate"] = calculate_range(learning_rates)
    config["optimization"]["n_layers"] = calculate_range(n_layers)

    # Set boolean values based on analysis
    config["optimization"]["use_batch_norm"] = has_batch_norm
    config["optimization"]["use_diminishing_layer_units"] = has_diminishing_units
    config["optimization"]["use_feature_masking"] = has_feature_masking

    print("\n\nOptimization configuration set based on previous best trials:")
    print(config['optimization'])
