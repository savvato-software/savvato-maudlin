import pytest
import os
import json
import shutil
from keras.models import Sequential
from keras.layers import Dense
from maudlin_core.src.model.track_best_metric import TrackBestMetric  # Assuming the class is saved in TrackBestMetric.py

@pytest.fixture
def setup_test_environment():
    # Setup test environment
    log_dir = "test_logs"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    yield log_dir
    # Cleanup after tests
    shutil.rmtree(log_dir)

@pytest.fixture
def sample_model():
    # Create a simple Keras model for testing
    model = Sequential([
        Dense(10, input_dim=5, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def test_track_best_metric_initialization(setup_test_environment):
    log_dir = setup_test_environment
    callback = TrackBestMetric(metric_names=['mae'], log_dir=log_dir)

    assert callback.metric_names == ['mae']
    assert 'mae' in callback.best_values
    assert callback.best_values['mae'] == float('inf')


def test_track_best_metric_logging(setup_test_environment, sample_model):
    log_dir = setup_test_environment
    callback = TrackBestMetric(metric_names=['mae'], log_dir=log_dir)

    # Simulate epochs
    logs_epoch_1 = {'mae': 0.3}
    logs_epoch_2 = {'mae': 0.2}

    callback.on_epoch_end(0, logs=logs_epoch_1)
    callback.on_epoch_end(1, logs=logs_epoch_2)

    # Verify best metrics
    with open(os.path.join(log_dir, "best_metrics.json"), "r") as f:
        data = json.load(f)

    assert 'mae' in data
    assert data['mae']['best_value'] == 0.2
    assert data['mae']['epoch'] == 2


def test_on_train_end(setup_test_environment, sample_model, capsys):
    log_dir = setup_test_environment
    callback = TrackBestMetric(metric_names=['mae'], log_dir=log_dir)

    logs_epoch_1 = {'mae': 0.3}
    logs_epoch_2 = {'mae': 0.2}

    callback.on_epoch_end(0, logs=logs_epoch_1)
    callback.on_epoch_end(1, logs=logs_epoch_2)
    callback.on_train_end()

    captured = capsys.readouterr()
    assert "Training complete. Best metrics:" in captured.out
    assert "mae = 0.2000" in captured.out
