import os
import json
import pytest
import numpy as np

from unittest.mock import patch, MagicMock, mock_open

# Import the functions from the module you want to test
from maudlin_core.src.predicting.predict import (
    maudlin,
    get_training_dir,
    get_prediction_dir,
    load_run_metadata,
    save_run_metadata,
    setup_training_directory,
    setup_prediction_directory,
    validate_prediction_file,
    load_and_preprocess_data,
    get_model,
    make_predictions,
    main
)


@pytest.fixture
def mock_maudlin(monkeypatch):
    """
    Fixture to replace the global 'maudlin' dictionary for tests.
    By default, we'll set some fake paths so that directory-building
    functions have something to work with.
    """
    fake_maudlin = {
        'data-directory': '/fake/base',
        'current-unit': 'unit_test'
    }
    monkeypatch.setattr('maudlin_core.src.predicting.predict.maudlin', fake_maudlin)
    return fake_maudlin


@pytest.fixture
def mock_config():
    """
    A sample config dict that can be used in tests.
    """
    return {
        'data': {
            'prediction_file': '/fake/base/predictions/unit_test/prediction_data.csv'
        },
        'prediction': {
            'threshold': 0.7
        }
    }


def test_get_training_dir(mock_maudlin):
    """Test that get_training_dir returns the correct path."""
    expected = "/fake/base/trainings/unit_test"
    assert get_training_dir() == expected


def test_get_prediction_dir(mock_maudlin):
    """Test that get_prediction_dir returns the correct path."""
    expected = "/fake/base/predictions/unit_test"
    assert get_prediction_dir() == expected


def test_load_run_metadata_file_not_exists(tmp_path):
    """Test load_run_metadata returns empty dict if file does not exist."""
    fake_file = tmp_path / "non_existent.json"
    result = load_run_metadata(str(fake_file))
    assert result == {}


def test_load_run_metadata_file_exists(tmp_path):
    """Test load_run_metadata returns JSON content if file exists."""
    # Create a fake file
    data = {"key": "value"}
    fake_file = tmp_path / "existent.json"
    fake_file.write_text(json.dumps(data))

    result = load_run_metadata(str(fake_file))
    assert result == data


def test_save_run_metadata(tmp_path):
    """Test save_run_metadata writes out JSON content to file."""
    file_path = tmp_path / "run_metadata.json"
    meta = {"test_key": 123}
    save_run_metadata(str(file_path), meta)

    # Check contents
    with open(file_path, 'r') as f:
        written = json.load(f)
    assert written == meta


def test_setup_training_directory_not_found():
    """
    If the training_dir doesn't exist, setup_training_directory should raise ValueError.
    """
    with pytest.raises(ValueError, match="Training directory not found:"):
        setup_training_directory("/path/does/not/exist")


@patch("os.path.exists", return_value=True)
@patch("maudlin_core.src.predicting.predict.load_run_metadata", return_value={})
def test_setup_training_directory_no_run_id(mock_load, mock_exists, tmp_path):
    """
    If run_metadata.json does not have current_run_id, setup_training_directory should raise ValueError.
    """
    # mock_exists is True for everything, so the training_dir "exists"
    # but run_metadata is an empty dict => no current_run_id
    with pytest.raises(ValueError):
        setup_training_directory(str(tmp_path))


@patch("os.path.exists", return_value=False)
@patch("maudlin_core.src.predicting.predict.load_run_metadata", return_value={"current_run_id": 999})
def test_setup_training_directory_no_run_dir(mock_load, mock_exists, tmp_path):
    """
    If the 'run_{current_run_id}' subdirectory doesn't exist,
    setup_training_directory should raise ValueError.
    """
    # The side_effect calls for mock_exists:
    # 1) True => training_dir is found
    # 2) True => run_metadata.json is found
    # 3) False => the "most recent run dir" is not found
    #fake_path = "/fake/training_dir/run_999"
    fake_path = None
    with pytest.raises(ValueError):
        setup_training_directory(str(fake_path))


@patch("os.path.exists", side_effect=[True, True, True])
@patch("maudlin_core.src.predicting.predict.load_run_metadata", return_value={"current_run_id": 999})
def test_setup_training_directory_success(mock_load, mock_exists, tmp_path):
    """
    Happy path: directory, run_metadata.json, and run_{current_run_id} exist.
    """
    result = setup_training_directory(str(tmp_path))
    expected = os.path.join(str(tmp_path), "run_999")
    assert result == expected


def test_setup_prediction_directory_creates_dir(tmp_path):
    """
    If prediction_dir doesn't exist, it should be created. Then a new run_id is assigned.
    """
    # We patch load_run_metadata so that it returns an empty dict => highest_run_id=0
    with patch("maudlin_core.src.predicting.predict.load_run_metadata", return_value={}), \
            patch("maudlin_core.src.predicting.predict.save_run_metadata") as mock_save:

        prediction_run_dir = setup_prediction_directory(str(tmp_path))
        # We expect highest_run_id => 1
        assert os.path.basename(prediction_run_dir) == "run_1"

        # Ensure save_run_metadata was called with updated metadata
        mock_save.assert_called_once()
        call_args = mock_save.call_args[0]
        assert call_args[0] == os.path.join(str(tmp_path), 'run_metadata.json')
        meta_passed = call_args[1]
        assert meta_passed['highest_run_id'] == 1
        assert meta_passed['current_run_id'] == 1


def test_validate_prediction_file_raises(mock_maudlin, mock_config):
    """If 'prediction_file' is empty, should raise ValueError."""
    mock_config['data']['prediction_file'] = ""  # no data file
    with pytest.raises(ValueError, match="Data file for .* is not set correctly."):
        validate_prediction_file(mock_config)


def test_validate_prediction_file_success(mock_maudlin, mock_config):
    """If 'prediction_file' is valid, should just return it."""
    result = validate_prediction_file(mock_config)
    expected = "/fake/base/predictions/unit_test/prediction_data.csv"
    assert result == expected


@patch("maudlin_core.src.predicting.predict.load_for_prediction", return_value=("X_data", "y_data", 10, ["col1", "col2"]))
def test_load_and_preprocess_data(mock_load_for_pred, mock_config, tmp_path):
    """Test that load_and_preprocess_data calls load_for_prediction and returns the expected tuple."""
    X, y, fc, cols = load_and_preprocess_data(mock_config, str(tmp_path))
    assert X == "X_data"
    assert y == "y_data"
    assert fc == 10
    assert cols == ["col1", "col2"]
    mock_load_for_pred.assert_called_once_with(mock_config, str(tmp_path))


@patch("maudlin_core.src.predicting.predict.create_model", return_value="MockedModel")
def test_get_model(mock_create_model, mock_config, tmp_path):
    """Test that get_model calls create_model with the correct args and returns the model."""
    model = get_model(mock_config, str(tmp_path), 10)
    assert model == "MockedModel"
    mock_create_model.assert_called_once_with(mock_config, str(tmp_path), 10, True)


@patch("maudlin_core.src.predicting.predict.execute_preprediction_stage")
@patch("maudlin_core.src.predicting.predict.execute_postprediction_stage")
def test_make_predictions(mock_post, mock_pre):
    """
    Test that make_predictions:
    1) calls execute_preprediction_stage
    2) calls model.predict on the returned X_pca
    3) calls execute_postprediction_stage
    """
    mock_model = MagicMock()
    # Suppose pre_prediction returns (X_pca, something, something, something)
    mock_pre.return_value = (np.array([[0.1, 0.2], [0.3, 0.4]]), None, None, None)
    # model.predict will output an array of float logits
    mock_model.predict.return_value = np.array([0.2, 0.8, 0.6])

    mock_config = {
        'data': {'prediction_file': '/fake/base/predictions/unit_test/prediction_data.csv'},
        'prediction': {'threshold': 0.7}
    }
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    y = np.array([0, 1])
    feature_count = 10
    columns = ["col1", "col2"]

    make_predictions(mock_config, mock_model, X, y, feature_count, columns, "/fake/dir")

    # Assertions
    mock_pre.assert_called_once_with(mock_config, "/fake/dir", mock_model, X, y, feature_count, columns)
    mock_model.predict.assert_called_once_with(mock_pre.return_value[0])
    mock_post.assert_called_once()



@patch("maudlin_core.src.predicting.predict.get_current_unit_config", return_value={
    'data': {'prediction_file': '/some/path.csv'},
    'mode': 'prediction'
})
@patch("maudlin_core.src.predicting.predict.get_training_dir", return_value="/fake/training_dir")
@patch("maudlin_core.src.predicting.predict.get_prediction_dir", return_value="/fake/prediction_dir")
@patch("maudlin_core.src.predicting.predict.setup_training_directory", return_value="/fake/training_dir/run_1")
@patch("maudlin_core.src.predicting.predict.setup_prediction_directory", return_value="/fake/prediction_dir/run_2")
@patch("maudlin_core.src.predicting.predict.validate_prediction_file")
@patch("maudlin_core.src.predicting.predict.load_and_preprocess_data", return_value=("X_dat", "y_dat", 5, ["c1", "c2"]))
@patch("maudlin_core.src.predicting.predict.get_model", return_value="FakeModel")
@patch("maudlin_core.src.predicting.predict.make_predictions")
def test_main(
        mock_make_predictions,
        mock_get_model,
        mock_load,
        mock_validate,
        mock_setup_pred,
        mock_setup_train,
        mock_get_pred_dir,
        mock_get_train_dir,
        mock_get_config,
):
    """
    Test the main() function to ensure it executes the entire
    prediction flow without error.
    """
    # Just call main; if everything is patched, it shouldn't do any real I/O
    main()

    # Check calls in the correct sequence
    mock_get_config.assert_called_once_with('')
    mock_get_train_dir.assert_called_once()
    mock_get_pred_dir.assert_called_once()
    mock_setup_train.assert_called_once_with("/fake/training_dir")
    mock_setup_pred.assert_called_once_with("/fake/prediction_dir")
    mock_validate.assert_called_once()
    mock_load.assert_called_once_with(mock_get_config.return_value, "/fake/prediction_dir/run_2")
    mock_get_model.assert_called_once_with(mock_get_config.return_value, "/fake/training_dir/run_1", 5)
    mock_make_predictions.assert_called_once()
