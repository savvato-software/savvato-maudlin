from maudlin_core.src.lib.framework.maudlin_unit_config import get_current_unit_config
from maudlin_core.src.lib.framework.maudlin import load_maudlin_data
from maudlin_core.src.common.data_preparation_manager import DataPreparationManager
from maudlin_core.src.model.adaptive_learning_rate import AdaptiveLearningRate
from maudlin_core.src.model.early_stopping_top_n import EarlyStoppingTopN
from maudlin_core.src.lib.framework.stage_functions.pre_optimization_function import execute_preoptimization_stage
from maudlin_core.src.lib.framework.stage_functions.post_optimization_function import execute_postoptimization_stage
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score

import optuna
import json

def main():
    data_dir = None

    config = get_current_unit_config()
    config['mode'] = 'optimize'
    maudlin = load_maudlin_data()

    # Setup directories
    # data_dir, run_id, parent_run_id = initialize_training_run_directory(maudlin)
    # config['run_id'] = run_id
    #config['parent_run_id'] = parent_run_id

    best_trials = []
    trial_configs = {}

    data_preparation_manager = DataPreparationManager(config, data_dir)

    X_train, y_train, X_test, y_test, X_val, y_val = data_preparation_manager.load_and_prepare_data()

    execute_preoptimization_stage(config)

    def objective(trial):
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
                patience=3,
                delta=0.001,
                top_n=3,
                patience_credit=0.25,
                strike_tolerance=0.04,
                trend_boost=0.8
            )
        ]

        update_config_with_suggested_model_architecture(trial, config)

        # print()
        # for item in best_trials:
        #     print(json.dumps(item, indent=None, separators=(", ", ": ")))
        # print()

        # TODO Implement feature masking

        print()
        print("Trial Number: ", trial.number)
        print("Trial Params: ", trial.params)
        print()

        model = data_preparation_manager.setup_model(X_train.shape[-1])
        _, model = data_preparation_manager.train_model(callbacks, X_train, y_train, X_val, y_val)

        # think this really is posttraining, not yet postoptimization
        y_preds = execute_postoptimization_stage(config, data_dir, data_preparation_manager.model, X_train, y_train, X_test, y_test)

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

        trial_configs[trial.number] = trial.params.copy()

        # Return the score for Optuna optimization
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=75)

    # for each best trial, display the trial_config
    for trial in best_trials:
        print("--------")
        print("Trial Number: ", trial.get('trial'))
        print(json.dumps(trial_configs[int(trial.get('trial'))], indent=4))
        print("--------")

    print("Best Params: ", study.best_params)

import yaml

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
    n_layers = trial.suggest_int('n_layers', 3, 4)
    initial_units = trial.suggest_int('initial_units', 45, 96)
    dropout = trial.suggest_float('dropout', 0.15, 0.3)
    learning_rate = trial.suggest_float('lr', 0.0000001, 0.001, log=True)
    optimizer = trial.suggest_categorical('optimizer', ['adam'])

    # TODO put these in the config ^^^

    # Define the architecture dynamically
    model_architecture = []
    units = initial_units

    for i in range(n_layers):
        model_architecture.append({
            'layer_type': 'Dense',
            'units': units,
            'activation': 'relu'
        })
        model_architecture.append({
            'layer_type': 'Dropout',
            'rate': dropout
        })

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

if __name__ == "__main__":
    main()
