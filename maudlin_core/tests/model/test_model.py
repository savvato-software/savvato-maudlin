import pytest
import os
import shutil
from tensorflow.keras.optimizers import Adam
from maudlin_core.src.model.model import create_model, generate_model_file_name

@pytest.fixture(scope="module")
def setup_test_environment():
    # Test directory setup
    test_dir = "./test_model_dir"
    os.makedirs(test_dir, exist_ok=True)
    yield test_dir
    shutil.rmtree(test_dir)

@pytest.fixture
def mock_config():
    return {
        'model_architecture': [
            {'layer_type': 'Dense', 'units': 32, 'activation': 'relu'},
            {'layer_type': 'Dropout', 'rate': 0.5},
            {'layer_type': 'Dense', 'units': 1, 'activation': 'sigmoid'}
        ],
        'learning_rate': 0.001,
        'metrics': ['accuracy'],
        'loss_function': 'binary_crossentropy',
        'optimizer': 'adam'
    }

@pytest.fixture
def mock_feature_count():
    return 10

# Test model file name generation
def test_generate_model_file_name(mock_config, setup_test_environment):
    test_dir = setup_test_environment
    model_file_name = generate_model_file_name(mock_config, test_dir)
    assert model_file_name.startswith(test_dir)
    assert model_file_name.endswith(".keras")

# Test model creation (new)
def test_create_new_model(mock_config, setup_test_environment, mock_feature_count):
    test_dir = setup_test_environment

    # Ensure the model file doesn't already exist
    model_file = generate_model_file_name(mock_config, test_dir)
    if os.path.exists(model_file):
        os.remove(model_file)

    model = create_model(mock_config, test_dir, mock_feature_count)

    # Check model layers
    assert len(model.layers) == 3  # 3 defined layers
    assert model.input_shape == (None, mock_feature_count)

    # Check optimizer
    optimizer = model.optimizer
    assert isinstance(optimizer, Adam)
    assert optimizer.learning_rate.numpy() == pytest.approx(0.001)

    # Check loss function
    assert model.loss == 'binary_crossentropy'


# Test loading existing model
def test_load_existing_model(mock_config, setup_test_environment, mock_feature_count):
    test_dir = setup_test_environment
    model_file = generate_model_file_name(mock_config, test_dir)

    # Create and save the model first
    create_model(mock_config, test_dir, mock_feature_count)

    # Load existing model
    model = create_model(mock_config, test_dir, mock_feature_count)

    # Check that it loads correctly
    assert len(model.layers) == 3
    assert model.input_shape == (None, mock_feature_count)

# Test failure when model file doesn't exist
def test_fail_if_not_existing(mock_config, setup_test_environment, mock_feature_count):
    test_dir = setup_test_environment
    model_file = generate_model_file_name(mock_config, test_dir)

    # Ensure model file doesn't exist
    if os.path.exists(model_file):
        os.remove(model_file)

    with pytest.raises(ValueError, match="The model does not exist"):
        create_model(mock_config, test_dir, mock_feature_count, fail_if_not_existing=True)
