import os
import pytest
import json
import shutil
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.model_selection import train_test_split

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from maudlin_core.src.training.batch import initialize_training_run_directory, signal_handler, create_model

@pytest.fixture
def setup_temp_dir(tmp_path):
    """Fixture to set up a temporary directory for tests."""
    test_dir = tmp_path / "maudlin_test"
    test_dir.mkdir()
    return test_dir

@pytest.fixture
def mock_maudlin_data():
    """Mock data for maudlin configuration."""
    return {
        'data-directory': '/tmp/maudlin',
        'current-unit': 'test_unit'
    }

@pytest.fixture
def mock_config():
    """Mock configuration data."""
    return {
        'data': {'training_file': 'mock_data.csv'},
        'timesteps': 10,
        'batch_size': 32,
        'epochs': 5,
        'metrics': ['mae'],
        'loss_function': 'mse',
        'model_architecture': [
            {'layer_type': 'LSTM', 'units': 64, 'return_sequences': True},
            {'layer_type': 'LSTM', 'units': 32, 'return_sequences': False},
            {'layer_type': 'Dense', 'units': 1, 'activation': 'sigmoid'}
        ],
        'training': {
            'adaptive_learning_rate': {
                'patience': 5,
                'factor': 0.85,
                'min-lr': 1e-6
            }
        },
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'pre_training': {
            'diagrams': ['oversampling', 'correlation_matrix', 'boxplot'],
            'oversampling': {
                'calculate_long_running_diagrams': False,
                'console_logging': True,
                'enabled': True,
                'k_neighbors': 3,
                'method': 'adasyn',
                'n_jobs': -1,
                'random_state': 42,
                'run_pca_before': False,
                'sampling_strategy': 0.8
            }
        }
    }

# Test: Initialize training run directory
def test_initialize_training_run_directory(setup_temp_dir, mock_maudlin_data):
    maudlin = mock_maudlin_data
    maudlin['data-directory'] = str(setup_temp_dir)

    # Create first run
    data_dir, run_id, prev_run_id = initialize_training_run_directory(maudlin)
    assert os.path.exists(data_dir)
    assert run_id == 1
    assert prev_run_id == 0

    # Create second run
    data_dir_2, run_id_2, prev_run_id_2 = initialize_training_run_directory(maudlin)
    assert os.path.exists(data_dir_2)
    assert run_id_2 == 2
    assert prev_run_id_2 == 1

# Test: Data split
def test_data_split():
    # Generate mock data
    X = np.random.rand(100, 10)
    y = np.random.rand(100, 1)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

    # Assert sizes
    assert X_train.shape[0] == 70
    assert X_val.shape[0] == 15
    assert X_test.shape[0] == 15

# Test: Model creation
def test_model_creation(mock_config):
    config = mock_config
    feature_count = 10

    # Mock model creation
    model = create_model(config, '/tmp', feature_count)

    assert isinstance(model, Sequential)
    assert len(model.layers) > 0



if __name__ == '__main__':
    pytest.main()
