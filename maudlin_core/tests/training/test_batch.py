import os
import json
import pytest
import signal
import sys
import shutil
import numpy as np

from unittest.mock import patch, MagicMock, mock_open, call
from argparse import Namespace

# Adjust these imports to match your actual code's structure:
from maudlin_core.src.lib.framework.maudlin import load_maudlin_data, get_current_unit_properties
from maudlin_core.src.lib.framework.maudlin_unit_config import get_current_unit_config
from maudlin_core.src.lib.data_loading.training import load_for_training
from maudlin_core.src.lib.framework.stage_functions.pre_training_function import execute_pretraining_stage
from maudlin_core.src.lib.framework.stage_functions.post_training_function import execute_posttraining_stage
from maudlin_core.src.model.model import create_model, generate_model_file_name
from maudlin_core.src.model.track_best_metric import TrackBestMetric
from maudlin_core.src.model.adaptive_learning_rate import AdaptiveLearningRate

# === The target code under test ===
# If your file is named differently or in a different path, change accordingly:
from maudlin_core.src.training.batch import (
    TrainingManager,
    initialize_training_run_directory,
    setup_signal_handler,
    run_batch_training
)


# --------------------------------------------------------------------------------
#                             TrainingManager Tests
# --------------------------------------------------------------------------------

@pytest.fixture
def sample_config():
    """Returns a minimal config dict for testing."""
    return {
        "data": {
            "training_file": "test_data.csv"
        },
        "pre_training": {
            "diagrams": ["oversampling", "correlation_matrix", "boxplot"],
            "oversampling": {
                "enabled": True,
                "method": "adasyn",
                "k_neighbors": 3,
                "sampling_strategy": 0.8,
                "random_state": 42,
                "run_pca_before": False
            }
        },
        "post_training": {
            "diagrams": ["confusion_matrix", "precision_recall_curve", "roc_curve"]
        },
        "learning_rate": 0.001,
        "loss_function": "mean_squared_error",
        "optimizer": "adam",
        "model_architecture": [
            {"layer_type": "Dense", "units": 256, "activation": "relu"},
            {"layer_type": "BatchNormalization"},
            {"layer_type": "Dense", "units": 128, "activation": "relu"},
            {"layer_type": "BatchNormalization"},
            {"layer_type": "Dropout", "rate": 0.2},
            {"layer_type": "Dense", "units": 1, "activation": "sigmoid"}
        ],        
        "training": {
            "adaptive_learning_rate": {
                "factor": 0.88,
                "min-lr": 1e-6,
                "patience": 5
            }
        },
        "epochs": 5,
        "batch_size": 32,
        "metrics": ["mae"],
        "classes": [
            {"label": 0, "weight": 1.0},
            {"label": 1, "weight": 2.0},
        ],
    }

@pytest.fixture
def training_manager(sample_config, tmp_path):
    """Instantiate a TrainingManager with mock config and a temp directory."""
    data_dir = str(tmp_path / "train_dir")
    os.makedirs(data_dir, exist_ok=True)
    return TrainingManager(config=sample_config, data_dir=data_dir)


def test_training_manager_init(training_manager, sample_config):
    """Test that the TrainingManager is initialized properly."""
    assert training_manager.config == sample_config
    assert os.path.exists(training_manager.data_dir)
    assert training_manager.model is None
    assert training_manager.model_file is None


@patch("maudlin_core.src.training.batch.load_for_training", 
       return_value=(np.array([[i] for i in range(10)]),  # X has 10 samples
                     np.arange(10),                      # y has 10 samples
                     10,                                 # feature_count
                     ["col1"]))
@patch("maudlin_core.src.training.batch.execute_pretraining_stage")
@patch("maudlin_core.src.training.batch.train_test_split")
def test_load_and_prepare_data(mock_split, mock_pretraining, mock_load, training_manager):
    """
    Test that load_and_prepare_data loads data, splits it, 
    and calls the pretraining stage.
    """
    X_mock = np.array([[1], [2], [3]])
    y_mock = np.array([1, 2, 3])

    # Mock train_test_split behavior:
    # For the first split, it returns X_train, X_temp, y_train, y_temp
    # For the second, it returns X_val, X_test, y_val, y_test
    mock_split.side_effect = [
        (X_mock[:2], X_mock[2:], y_mock[:2], y_mock[2:]),  # 2-1 split
        (X_mock[2:], X_mock[2:], y_mock[2:], y_mock[2:])   # Fake 1-1 split
    ]

    # Mock pretraining stage
    mock_pretraining.return_value = (
        X_mock[:2], y_mock[:2], 
        X_mock[2:], y_mock[2:], 
        X_mock[2:], y_mock[2:]
    )

    X_train, y_train, X_test, y_test, X_val, y_val = training_manager.load_and_prepare_data()

    # Check calls
    mock_load.assert_called_once_with(training_manager.config, training_manager.data_dir)
    assert mock_split.call_count == 2
    mock_pretraining.assert_called_once()

    # Verify final returned shapes match pretraining stage return
    np.testing.assert_array_equal(X_train, X_mock[:2])
    np.testing.assert_array_equal(X_test, X_mock[2:])
    np.testing.assert_array_equal(X_val, X_mock[2:])
    np.testing.assert_array_equal(y_train, y_mock[:2])
    np.testing.assert_array_equal(y_test, y_mock[2:])
    np.testing.assert_array_equal(y_val, y_mock[2:])


@patch("maudlin_core.src.training.batch.create_model")
@patch("maudlin_core.src.training.batch.generate_model_file_name")
def test_setup_model(mock_gen_name, mock_create, training_manager):
    """Test setup_model calls create_model and generate_model_file_name properly."""
    mock_gen_name.return_value = "model_file_path.h5"
    fake_model = MagicMock()
    mock_create.return_value = fake_model

    input_shape = 10
    model = training_manager.setup_model(input_shape)

    mock_create.assert_called_once_with(training_manager.config, training_manager.data_dir, input_shape)
    mock_gen_name.assert_called_once_with(training_manager.config, training_manager.data_dir)
    assert model == fake_model
    assert training_manager.model_file == "model_file_path.h5"


def test_get_class_weights(training_manager):
    """Test get_class_weights returns the correct dictionary."""
    weights = training_manager.get_class_weights()
    assert weights == {0: 1.0, 1: 2.0}

    # Test scenario with no 'classes' in config
    training_manager.config.pop("classes")
    weights = training_manager.get_class_weights()
    assert weights == {}


@patch("maudlin_core.src.training.batch.TensorBoard")
@patch("maudlin_core.src.training.batch.AdaptiveLearningRate")
@patch("maudlin_core.src.training.batch.TrackBestMetric")
def test_setup_callbacks(mock_metric_tracker, mock_adaptive_lr, mock_tensorboard, training_manager):
    """Test setup_callbacks returns a proper list of callbacks."""
    # Change metrics for variety
    training_manager.config["metrics"] = ["mse"]
    cbs = training_manager.setup_callbacks()

    # We expect 3 callbacks
    assert len(cbs) == 3
    # Check instantiations
    mock_adaptive_lr.assert_called_once()
    mock_metric_tracker.assert_called_once()
    mock_tensorboard.assert_called_once()


@patch.object(TrainingManager, "setup_callbacks", return_value=["mock_callbacks"])
@patch.object(TrainingManager, "get_class_weights", return_value={0: 1.0, 1: 2.0})
def test_train_model(mock_class_weights, mock_callbacks, training_manager):
    """Test train_model calls model.fit with the right arguments."""
    # Setup a fake model
    fake_model = MagicMock()
    training_manager.model = fake_model
    X_train, y_train = np.array([[1],[2]]), np.array([0,1])
    X_val, y_val = np.array([[3],[4]]), np.array([0,1])

    training_manager.config["epochs"] = 5
    training_manager.config["batch_size"] = 16

    training_manager.train_model(X_train, y_train, X_val, y_val)

    # Check calls
    fake_model.fit.assert_called_once_with(
        X_train, y_train,
        epochs=5,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=["mock_callbacks"],
        class_weight={0: 1.0, 1: 2.0}
    )


def test_save_model_no_model(training_manager):
    """Test save_model with no model or model_file set."""
    assert not training_manager.save_model(), "Should return False if no model or model_file is set"


def test_save_model_success(training_manager):
    """Test save_model with valid model and model file."""
    training_manager.model = MagicMock()
    training_manager.model_file = "mock_model_path.h5"
    assert training_manager.save_model(), "Should return True on successful save"
    training_manager.model.save.assert_called_once_with("mock_model_path.h5")


# --------------------------------------------------------------------------------
#                  initialize_training_run_directory Tests
# --------------------------------------------------------------------------------

@pytest.fixture
def mock_maudlin(tmp_path):
    """Creates a mock maudlin object with a custom data-directory."""
    return {
        "data-directory": str(tmp_path),
        "current-unit": "test_unit"
    }


@pytest.mark.parametrize("file_exists", [True, False])
def test_initialize_training_run_directory(file_exists, mock_maudlin):
    """Test initialize_training_run_directory with/without existing run_metadata.json."""
    unit_dir = os.path.join(mock_maudlin["data-directory"], "trainings", mock_maudlin["current-unit"])
    os.makedirs(unit_dir, exist_ok=True)
    run_metadata_path = os.path.join(unit_dir, "run_metadata.json")

    if file_exists:
        existing_data = {"highest_run_id": 10, "current_run_id": 9}
        with open(run_metadata_path, "w") as f:
            json.dump(existing_data, f)

    data_dir, current_run_id, prev_run_id = initialize_training_run_directory(mock_maudlin)

    # Check that the run_metadata file was created or updated
    assert os.path.exists(run_metadata_path)

    with open(run_metadata_path, "r") as f:
        data = json.load(f)

    # If the file didn't exist before, default was 0
    if not file_exists:
        assert data["highest_run_id"] == 1
        assert data["current_run_id"] == 1
        assert prev_run_id == 0
        assert current_run_id == 1
    else:
        # Should increment from 10 to 11
        assert data["highest_run_id"] == 11
        assert data["current_run_id"] == 11
        assert prev_run_id == 9
        assert current_run_id == 11

    # Check that data_dir was created
    assert os.path.exists(data_dir)


# --------------------------------------------------------------------------------
#                     setup_signal_handler Tests
# --------------------------------------------------------------------------------

@patch("signal.signal")
def test_setup_signal_handler(mock_signal, training_manager):
    """Test that setup_signal_handler sets up the signal properly."""
    setup_signal_handler(training_manager)
    mock_signal.assert_called_once()
    # Check that the handler is a function
    args, kwargs = mock_signal.call_args
    assert args[0] == signal.SIGINT
    assert callable(args[1])


@patch("signal.signal")
def test_signal_handler(mock_signal, training_manager, capsys):
    """
    Test that the signal handler calls training_manager.save_model() 
    and sys.exit(0) upon Ctrl+C.
    """
    setup_signal_handler(training_manager)
    handler = mock_signal.call_args[0][1]

    # Mock out sys.exit
    with patch("sys.exit") as mock_exit:
        # Mock out the training_manager.save_model to print something
        with patch.object(training_manager, "save_model", return_value=True) as mock_save:
            handler(signal.SIGINT, None)
            mock_save.assert_called_once()
            mock_exit.assert_called_once_with(0)

    captured = capsys.readouterr()
    assert "Interrupt received! Saving the model" in captured.out


# --------------------------------------------------------------------------------
#                      run_batch_training Tests
# --------------------------------------------------------------------------------

@pytest.fixture
def mock_cli_args(tmp_path):
    """Provide a mock CLI args for run_batch_training."""
    return [str(tmp_path / "fake_training_run")]


@patch("argparse.ArgumentParser.parse_args")
@patch("maudlin_core.src.lib.framework.maudlin_unit_config.get_current_unit_config", return_value={"some": "config"})
@patch("maudlin_core.src.training.batch.get_current_unit_properties", return_value={"config-path": "some_config.yaml"})
@patch("maudlin_core.src.training.batch.load_maudlin_data", return_value={"data-directory": "/tmp", "current-unit": "mock_unit"})
@patch("shutil.copy")
@patch("maudlin_core.src.training.batch.initialize_training_run_directory", return_value=("/tmp/run_1", 1, 0))
@patch("maudlin_core.src.training.batch.TrainingManager")
@patch("maudlin_core.src.training.batch.setup_signal_handler")
@patch("maudlin_core.src.training.batch.execute_posttraining_stage")
def test_run_batch_training_success(
    mock_posttraining,
    mock_signal_handler,
    mock_training_manager_cls,
    mock_init_dir,
    mock_copy,
    mock_load_maudlin_data,
    mock_get_props,
    mock_get_config,
    mock_parse_args
):
    """Test run_batch_training with normal successful flow."""
    # Mock parse_args
    args = Namespace(training_run_path=None)
    mock_parse_args.return_value = args

    # Mock a training_manager instance
    mock_tm_instance = MagicMock()
    mock_tm_instance.load_and_prepare_data.return_value = (
        np.array([]), np.array([]),
        np.array([]), np.array([]),
        np.array([]), np.array([])
    )
    
    mock_training_manager_cls.return_value = mock_tm_instance    

    run_batch_training()

    # We expect the manager's calls
    mock_tm_instance.load_and_prepare_data.assert_called_once()
    mock_tm_instance.setup_model.assert_called_once()
    mock_tm_instance.train_model.assert_called_once()
    mock_tm_instance.save_model.assert_called_once()
    mock_posttraining.assert_called_once()


@patch("argparse.ArgumentParser.parse_args")
@patch("maudlin_core.src.lib.framework.maudlin_unit_config.get_current_unit_config", return_value={"some": "config"})
@patch("maudlin_core.src.training.batch.get_current_unit_properties", return_value={"config-path": "some_config.yaml"})
@patch("maudlin_core.src.training.batch.load_maudlin_data", return_value={"data-directory": "/tmp", "current-unit": "mock_unit"})
@patch("maudlin_core.src.training.batch.initialize_training_run_directory", return_value=("/tmp/run_2", 2, 1))
@patch("maudlin_core.src.training.batch.TrainingManager")
def test_run_batch_training_interrupt(
    mock_training_manager_cls,
    mock_init_dir,
    mock_load_maudlin_data,
    mock_get_props,
    mock_get_config,
    mock_parse_args,
):
    """Test run_batch_training when a KeyboardInterrupt is raised."""
    args = Namespace(training_run_path=None)
    mock_parse_args.return_value = args

    mock_tm_instance = MagicMock()
    mock_tm_instance.load_and_prepare_data.return_value = (
        np.array([]), np.array([]),
        np.array([]), np.array([]),
        np.array([]), np.array([])
    )
    
    mock_training_manager_cls.return_value = mock_tm_instance

    # Force an interrupt during load_and_prepare_data
    mock_tm_instance.load_and_prepare_data.side_effect = KeyboardInterrupt()

    with patch("builtins.print") as mock_print:
        run_batch_training()

    # We expect that upon KeyboardInterrupt, the model is saved and we see an interrupt message
    mock_tm_instance.save_model.assert_called_once()
    interrupt_calls = [c for c in mock_print.call_args_list if "Training interrupted." in str(c)]
    assert len(interrupt_calls) == 1, "Expected an interrupt message printed."


@patch("maudlin_core.src.training.batch.TrainingManager")
@patch("maudlin_core.src.training.batch.initialize_training_run_directory", return_value=("/tmp/run_2", 2, 1))
@patch("maudlin_core.src.training.batch.load_maudlin_data", return_value={"data-directory": "/tmp", "current-unit": "mock_unit"})
@patch("maudlin_core.src.training.batch.get_current_unit_properties", return_value={"config-path": "some_config.yaml"})
@patch("maudlin_core.src.lib.framework.maudlin_unit_config.get_current_unit_config", return_value={"some": "config"})
@patch("maudlin_core.src.training.batch.argparse.ArgumentParser")
def test_run_batch_training_with_cli_path(
    mock_parser_cls,
    mock_get_config,
    mock_get_props,
    mock_load_maudlin_data,
    mock_init_dir,
    mock_training_manager_cls,
):
    """Test run_batch_training when a training_run_path is explicitly passed."""
    # Create a concrete string value for training_run_path
    training_run_path = "/tmp/run_2/config.yaml"

    # Setup the ArgumentParser mock
    mock_parser = MagicMock()
    mock_parser_cls.return_value = mock_parser
    args = Namespace(training_run_path=training_run_path)
    mock_parser.parse_args.return_value = args

    # Mock a training_manager instance
    mock_tm_instance = MagicMock()
    # Set up return values for load_and_prepare_data
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    mock_tm_instance.load_and_prepare_data.return_value = (X, y, X, y, X, y)
    
    # Set up a mock model that returns numpy arrays for predict
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.3, 0.7])
    mock_tm_instance.model = mock_model
    
    mock_training_manager_cls.return_value = mock_tm_instance

    run_batch_training()

    # Verify the config file was copied
    mock_tm_instance.copy_config_file.assert_called_once_with(training_run_path)

    # Verify other expected calls
    mock_tm_instance.load_and_prepare_data.assert_called_once()
    mock_tm_instance.setup_model.assert_called_once()
    mock_tm_instance.train_model.assert_called_once()
    mock_tm_instance.save_model.assert_called_once()  
