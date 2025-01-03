from maudlin_core.src.model.model import create_model
from maudlin_core.src.lib.framework.maudlin_unit_config import get_current_unit_config
from maudlin_core.src.lib.framework.maudlin import load_maudlin_data
from maudlin_core.src.lib.framework.stage_functions.pre_prediction_function import execute_preprediction_stage
from maudlin_core.src.lib.framework.stage_functions.post_prediction_function import execute_postprediction_stage

from maudlin_core.src.lib.data_loading.prediction import load_for_prediction

import os
import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_config():
    return {
        'timesteps': 10,
        'data': {'prediction_file': 'test_file.csv'},
        'prediction': {'threshold': 0.5},
        'training_run_path': '/mock/training/run/path',
        'model_architecture': [
            {'layer_type': 'LSTM', 'units': 64, 'return_sequences': True},
            {'layer_type': 'LSTM', 'units': 32, 'return_sequences': False},
            {'layer_type': 'Dense', 'units': 1, 'activation': 'sigmoid'}
        ],
        'optimizer': 'adam',
        'loss_function': 'binary_crossentropy',
        'metrics': ['accuracy'],
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        'pca': {'console_logging': True, 'enabled': True, 'params': {'n_components': 4}},
        'pre_prediction': {'diagrams': []},
        'prediction': {'threshold': 0.5, 'perturbation': {'enabled': True}, 'features': [
                       {'name': 'age', 'type': 'int', 'range': [-10, 20], 'min': 0, 'max': 120},
                          {'name': 'balance', 'type': 'int', 'range': [-25, 100], 'min': 0},
                            {'name': 'housing', 'type': 'binary'},
                                {'name': 'loan', 'type': 'binary'}]},
        'post_prediction': {'diagrams': []}
    }

@pytest.fixture
def mock_maudlin():
    return {
        'data-directory': '/mock/data',
        'current-unit': 'test_unit'
    }

@pytest.fixture
def mock_dirs(tmpdir):
    train_dir = tmpdir.mkdir("trainings").mkdir("test_unit").mkdir("run_1")
    pred_dir = tmpdir.mkdir("predictions").mkdir("test_unit").mkdir("run_1")

    train_metadata = {'current_run_id': 1}
    pred_metadata = {'highest_run_id': 1, 'current_run_id': 1}

    with open(os.path.join(train_dir, 'run_metadata.json'), 'w') as file:
        json.dump(train_metadata, file)

    with open(os.path.join(pred_dir, 'run_metadata.json'), 'w') as file:
        json.dump(pred_metadata, file)

    return str(pred_dir), str(train_dir)

# Mock using the actual import path
@patch('test_predict.load_for_prediction')
@patch('maudlin_core.src.lib.framework.maudlin.load_maudlin_data')
@patch('maudlin_core.src.lib.framework.maudlin_unit_config.get_current_unit_config')
@patch('test_predict.create_model')
@patch('test_predict.execute_preprediction_stage')
@patch('test_predict.execute_postprediction_stage')
def test_prediction_pipeline(mock_post_pred, mock_pre_pred, mock_create_model, mock_get_config, mock_load_maudlin, mock_load_for_pred, mock_maudlin, mock_config, mock_dirs):
    # Mock inputs
    mock_load_maudlin.return_value = mock_maudlin
    mock_get_config.return_value = mock_config

    prediction_dir, training_dir = mock_dirs
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, size=(10,))
    feature_count = 5
    columns = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']

    mock_load_for_pred.return_value = (X, y, feature_count, columns)
    mock_model = MagicMock()
    mock_create_model.return_value = mock_model
    mock_pre_pred.return_value = (X, None, None, None)
    mock_model.predict.return_value = np.random.rand(10, 1)

    # Simulate script execution
    config = mock_config
    config['mode'] = 'prediction'

    # Check directory setup
    assert os.path.exists(prediction_dir)
    assert os.path.exists(training_dir)

    # Mock data loading
    X, y, feature_count, columns = load_for_prediction(config, prediction_dir)

    # Model creation
    model = create_model(config, training_dir, feature_count, True)

    # Pre-prediction stage
    X_pca, _, _, _ = execute_preprediction_stage(config, prediction_dir, model, X, y, feature_count, columns)

    # Predictions
    predictions = model.predict(X_pca)
    threshold = config['prediction']['threshold']
    y_preds = (predictions >= threshold).astype('int')

    # Post-prediction stage
    execute_postprediction_stage(config, prediction_dir, model, y_preds, y)

    # Assertions
    mock_load_for_pred.assert_called_once()
    mock_create_model.assert_called_once()
    mock_pre_pred.assert_called_once()
    mock_model.predict.assert_called_once_with(X_pca)
    mock_post_pred.assert_called_once()
    assert isinstance(y_preds, np.ndarray)
    assert len(y_preds) == len(y)
