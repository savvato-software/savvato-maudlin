from maudlin_core.src.lib.framework.maudlin_unit_config import get_current_unit_config
from maudlin_core.src.lib.framework.maudlin import load_maudlin_data, get_current_unit_name
from maudlin_core.src.common.data_preparation_manager import DataPreparationManager
from maudlin_core.src.model.adaptive_learning_rate import AdaptiveLearningRate
from maudlin_core.src.model.early_stopping_top_n import EarlyStoppingTopN
from maudlin_core.src.lib.framework.stage_functions.pre_optimization_function import execute_preoptimization_stage
from maudlin_core.src.lib.framework.stage_functions.post_optimization_function import execute_postoptimization_stage
from maudlin_core.src.lib.framework.maudlin import load_yaml_file, save_yaml_file
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from functools import partial

import os
import optuna
import json
import yaml
import shutil
import datetime
import argparse

def run_optimization():
    main()

def main(cli_args=None):
    """Main entry point for the optimization script."""
    # Parse CLI args
    parser = argparse.ArgumentParser(description="Optimize a model using Optuna.")
    parser.add_argument(
        "--trials", "-t",
        type=int,
        help="Number of trials to run.",
    )
    parser.add_argument(
        "optimization_run_path",
        nargs='?',
        default=None,
        help="Path to the optimization run directory (optional)."
    )
    parser.add_argument(
        "--use-existing-model", "-m",
        action='store_true',
        help="Use an existing model for training."
    )
    parser.add_argument(
        "--use-last-best-trials", "-lbt",
        action='store_true',
        help="Use the last best trials to constrain a new optimization run."
    )
    args = parser.parse_args(cli_args) if cli_args else parser.parse_args()

    data_dir = None

    config = get_current_unit_config(args.optimization_run_path)
    config['mode'] = 'optimize'
    config['optimization']['n_trials'] = args.trials

    rconfig = config.get('runtime')

    if not rconfig:
        rconfig = {}
        config['runtime'] = rconfig

    rconfig['use_existing_model'] = args.use_existing_model
    rconfig['use_last_best_trials'] = args.use_last_best_trials

    ## create a runtime section in the config, for runtime params.. hack, because we're saving the config, and this stuff is in there...

    if (rconfig['use_existing_model']):
        print("---- using EXISTING model")
    elif (rconfig['use_last_best_trials']):
        print("---- optimizing A NEW model using the last best trials")
    else:
        print("---- optimizing A NEW model from scratch")

    maudlin = load_maudlin_data()

    # Setup directories
    data_dir, unit_dir, run_id, parent_run_id = initialize_optimization_run_directory(maudlin, config)
    rconfig['run_id'] = run_id
    rconfig['parent_run_id'] = parent_run_id
    rconfig['current_unit_name'] = get_current_unit_name(maudlin)

    if ('base_optimization_run_id' not in rconfig) or (rconfig['use_last_best_trials']):
        rconfig['base_optimization_run_id'] = rconfig['run_id']

    oconfig = config['optimization']

    best_trials = []
    trial_configs = {}

    data_preparation_manager = DataPreparationManager(config, data_dir)

    X_train, y_train, X_test, y_test, X_val, y_val = data_preparation_manager.load_and_prepare_data()

    execute_preoptimization_stage(config, unit_dir)

    ctr = 0

    def objective(trial, ctr, X_train, y_train, X_test, y_test, X_val, y_val):
        ctr += 1

        X_train_masked = X_train.copy()
        X_test_masked = X_test.copy()
        X_val_masked = X_val.copy()

        if oconfig['use_feature_masking']:
            feature_mask = [trial.suggest_categorical(f'feature_{i}', [0, 1]) for i in range(X_train_masked.shape[1])]
            selected_features = [i for i in range(len(feature_mask)) if feature_mask[i] == 1]
            X_train_masked = X_train_masked[:, selected_features]
            X_test_masked = X_test_masked[:, selected_features]
            X_val_masked = X_val_masked[:, selected_features]

        # """Configure training callbacks"""
        metrics_to_track = config.get("metrics", ["mae"])
        alrconfig = config['training']['adaptive_learning_rate']

        callbacks = [
            AdaptiveLearningRate(
                metric_name=metrics_to_track[0],
                patience=alrconfig['patience'],
                factor=alrconfig['factor'],
                min_lr=alrconfig['min-lr']
            ),
            EarlyStoppingTopN(
                metric_name=metrics_to_track[0],
                trial=trial,
                best_trials=best_trials,
                patience=oconfig['patience'],
                delta=oconfig['delta'],
                top_n=oconfig['top_n'],
                patience_credit=oconfig['patience_credit'],
                strike_tolerance=oconfig['strike_tolerance'],
                trend_boost=oconfig['trend_boost'],
                lbpr_start_count=oconfig['lbpr_start_count']
            )
        ]

        update_config_with_suggested_model_architecture(trial, config)

        config['batch_size'] = trial.suggest_int('batch_size', oconfig['batch_size']['min'], oconfig['batch_size']['max'])

        print()
        print("Trial Number: ", trial.number, " | ", oconfig['n_trials'] - trial.number, "   ", ctr, " trials remaining")
        print("Trial Params: ", trial.params)
        print()

        config['epochs'] = 20
        model = data_preparation_manager.setup_model(X_train_masked.shape[-1])
        _, model = data_preparation_manager.train_model(callbacks, X_train_masked, y_train, X_val_masked, y_val)

        # think this really is posttraining, not yet postoptimization
        y_preds = execute_postoptimization_stage(config, data_dir, data_preparation_manager.model, X_train_masked, y_train, X_test_masked, y_test)

        # Evaluate based on the first metric listed in the config
        metrics = config['metrics']
        primary_metric = metrics[0].lower()  # First metric in config, normalized to lowercase

        # Compute the required metric
        if primary_metric == 'auc':
            score = roc_auc_score(y_test, y_preds)
        elif primary_metric == 'mae':
            score = mean_absolute_error(y_test, y_preds)
        elif primary_metric == 'accuracy':
            score = accuracy_score(y_test, y_preds)  # Assuming binary classification
        else:
            raise ValueError(f"Unsupported metric: {primary_metric}")

        obj = {
            "params": trial.params.copy(),
            # "values": trial.values.copy(),
            "model_architecture": config.get('model_architecture')
        }

        trial_configs[trial.number] = obj

        # Return the score for Optuna optimization
        return score

    study = get_study_object(config) # optuna.create_study(direction='maximize')
    objective_with_data = partial(objective, ctr=ctr, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, X_val=X_val, y_val=y_val)
    study.optimize(objective_with_data, n_trials=oconfig['n_trials'])

    # Write best trials to a file
    best_trials_file = os.path.join(data_dir, "best_trials.yaml")
    best_trial_data = []

    for trial in best_trials:
        trial_number = trial.get('trial')
        trial_info = {
            "trial_number": trial_number,
            "params": trial_configs[int(trial_number)]['params'],
            "trial_values": trial.get('values'),
            "model_architecture": trial_configs[int(trial_number)]['model_architecture'],
        }
        best_trial_data.append(trial_info)

        print("--------")
        print("Trial Number: ", trial_number)
        print(yaml.dump(trial_configs[int(trial_number)], indent=4))
        print("--------")

    # Save to JSON file
    with open(best_trials_file, 'w') as f:
        yaml.dump(best_trial_data, f, indent=4)

    print("Best Params: ", study.best_params)
    print()
    print("Finished at ", datetime.datetime.now())

def get_study_object(config):
    # Define the database file for SQLite
    db_path = "optuna_study.db"
    storage = f"sqlite:///{db_path}"

    study = None

    if config and config['runtime'] and config['runtime']['use_existing_model']:
        # Load the study from the SQLite database
        name = f"optuna_study_run_{config['runtime']['base_optimization_run_id']}"
        print(f"Loading existing study: {name}")
        study = optuna.load_study(study_name=name, storage=storage)
    else:
        # Create a study and connect it to the SQLite database
        name = f"optuna_study_run_{config['runtime']['run_id']}"
        print(f"Creating new study: {name}")
        study = optuna.create_study(study_name=name, storage=storage, direction="maximize")

    return study


def update_config_with_suggested_model_architecture(trial, base_config):
    """
    Generate a Maudlin-compatible YAML configuration file from Optuna hyperparameters.

    Args:
        trial (optuna.Trial): Current Optuna trial containing hyperparameters.
        base_config (dict): Base configuration with static settings.

    Returns:
        str: Path to the generated YAML file.
    """
    # Extract hyperparameters from the trial
    c = base_config['optimization']
    n_layers = trial.suggest_int('n_layers', c['n_layers']['min'], c['n_layers']['max'])
    initial_units = trial.suggest_int('initial_layer_units', c['initial_layer_units']['min'], c['initial_layer_units']['max'])
    dropout = trial.suggest_float('dropout', c['dropout']['min'], c['dropout']['max'])
    learning_rate = trial.suggest_float('lr', c['learning_rate']['min'], c['learning_rate']['max'], log=True)
    optimizer = trial.suggest_categorical('optimizer', c['optimizer'])
    activation = trial.suggest_categorical('activation', c['activation'])

    use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False]) if c['use_batch_norm'] else False
    use_diminishing_layer_units = trial.suggest_categorical('use_diminishing_layer_units', [True, False]) if c['use_diminishing_layer_units'] else False

    # Define the architecture dynamically
    model_architecture = []
    units = initial_units

    for i in range(n_layers):
        model_architecture.append({
            'layer_type': 'Dense',
            'units': units,
            'activation': activation
        })

        if use_batch_norm:
            model_architecture.append({
                'layer_type': 'BatchNormalization'
            })

        model_architecture.append({
            'layer_type': 'Dropout',
            'rate': dropout
        })

        if use_diminishing_layer_units:
            units = max(units // 2, 4)

    # Add output layer
    model_architecture.append({
        'layer_type': 'Dense',
        'units': 1,
        'activation': 'sigmoid'  # Assuming binary classification
    })

    # Update base config
    base_config['model_architecture'] = model_architecture
    base_config['learning_rate'] = learning_rate
    base_config['optimizer'] = optimizer

    return base_config

## TODO: Move this to a common utility file, its in batch.py as well
def initialize_optimization_run_directory(maudlin, config):
    # Define paths
    unit_dir = os.path.join(maudlin['data-directory'], 'optimizations', maudlin['current-unit'])
    os.makedirs(unit_dir, exist_ok=True)

    # Counter file path
    counter_file = os.path.join(unit_dir, 'run_metadata.json')

    # Initialize metadata
    if os.path.exists(counter_file):
        with open(counter_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {'highest_run_id': 0, 'current_run_id': 0}

    prev_curr_run_id = metadata['current_run_id']

    # Increment the highest run ID
    metadata['highest_run_id'] += 1
    metadata['current_run_id'] = metadata['highest_run_id']

    # Write updated metadata back to the file
    with open(counter_file, 'w') as f:
        json.dump(metadata, f, indent=4)

    # Create directory for the current run
    data_dir = os.path.join(unit_dir, f"run_{metadata['current_run_id']}")
    os.makedirs(data_dir, exist_ok=True)

    # Copy config file to the current run directory
    prev_config_path = ''

    if config and config.get('runtime') and config.get('runtime').get('use_existing_model'):
        prev_config_path = os.path.join(unit_dir, f"run_{prev_curr_run_id}/config.yaml")
    else:
        prev_config_path = os.path.join(maudlin['data-directory'], 'configs', maudlin['current-unit'] + ".config.yaml")

    if not os.path.exists(prev_config_path):
        raise FileNotFoundError(f"Config file not found at {prev_config_path}")

    shutil.copy(prev_config_path, data_dir + "/config.yaml")

    # open the config file and update the run_id and parent_run_id
    c = load_yaml_file(data_dir + "/config.yaml")

    if 'runtime' not in c:
        c['runtime'] = {}

    c['runtime']['run_id'] = metadata['current_run_id']
    c['runtime']['parent_run_id'] = prev_curr_run_id
    save_yaml_file(c, data_dir + "/config.yaml")

    return data_dir, unit_dir, metadata['current_run_id'], prev_curr_run_id


if __name__ == "__main__":
    main()
