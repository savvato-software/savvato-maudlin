import os
import hashlib

from keras.src.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, RMSprop, Nadam
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout

from ..lib.framework.maudlin import load_maudlin_data, get_current_unit_properties, write_keras_filename_for_current_unit, get_unit_function_path

from ..lib.savvato_python_functions import load_function_from_file

# Generate a readable model file name based on architecture
def generate_model_file_name(config, data_dir):
    """
    Generate a unique filename for the model based on the architecture using a hash.
    """

#    rtn = get_current_unit_properties(_maudlin)['keras-filename']

 #   if not rtn:
    architecture_string = str(config['model_architecture']).encode('utf-8')
    architecture_hash = hashlib.md5(architecture_string).hexdigest()
    rtn = f"model_{architecture_hash}.keras"
#    rtn = write_keras_filename_for_current_unit(data_dir, rtn)
  #  else:
   #     rtn = _maudlin['data-directory'] + rtn

    return data_dir + "/" + rtn


def instantiate_model(config, feature_count):
    """
    Create a model based on the current unit's configuration.

    Args:
        config (dict): Model configuration.
        feature_count (int): Number of features in the input data.

    Returns:
        keras.models.Sequential: Configured model.
    """

    # Map of supported layer types to Keras classes
    layer_mapping = {
        'Conv1D': Conv1D,
        'MaxPooling1D': MaxPooling1D,
        'LSTM': LSTM,
        'Dense': Dense,
        'BatchNormalization': BatchNormalization,
        'Dropout': Dropout,
    }

    # Initialize the Sequential model
    model = Sequential()

    # Explicitly define the input shape using an Input layer
    if config.get('timesteps'):  # Time-series data
        model.add(Input(shape=(config['timesteps'], feature_count)))
    else:  # Non-time-series data
        model.add(Input(shape=(feature_count,)))

    # Add the remaining layers dynamically
    for i, layer_config in enumerate(config['model_architecture']):
        # Create a copy to avoid modifying the original configuration
        layer_config_copy = layer_config.copy()

        layer_type = layer_config_copy.pop('layer_type')

        # Add the layer to the model
        if layer_type in layer_mapping:
            model.add(layer_mapping[layer_type](**layer_config_copy))
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    return model


def get_optimizer(optimizer_name, learning_rate):
    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adam':
        return Adam(learning_rate=learning_rate)
    elif optimizer_name == 'nadam':
        return Nadam(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        return SGD(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        return RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'adagrad':
        return Adagrad(learning_rate=learning_rate)
    else:
        print(f"Unknown optimizer '{optimizer_name}', defaulting to 'adam'.")
        return Adam(learning_rate=learning_rate)


def load_existing_model(model_file, optimizer, loss_function, metrics):
    print(f"Loading existing model from {model_file}...")
    model = load_model(model_file)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
    return model


def create_new_model(config, feature_count, optimizer, loss_function, metrics):
    print(f"Creating a new model...")
    model = instantiate_model(config, feature_count)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
    return model

# Main function to handle both loading and creating a model
def create_model(config, data_dir, feature_count, fail_if_not_existing=False):
    maudlin = load_maudlin_data()
    if data_dir:
        MODEL_FILE = generate_model_file_name(config, data_dir)

    # Extract configuration details
    LEARNING_RATE = config['learning_rate']
    METRICS = config['metrics']
    LOSS_FUNCTION = config['loss_function']
    OPTIMIZER_NAME = config.get('optimizer', 'adam')

    # Get optimizer
    optimizer = get_optimizer(OPTIMIZER_NAME, LEARNING_RATE)

    # Load or create model
    if data_dir and os.path.exists(MODEL_FILE):
        model = load_existing_model(MODEL_FILE, optimizer, LOSS_FUNCTION, METRICS)
    elif fail_if_not_existing:
        raise ValueError(f"The model does not exist at {MODEL_FILE}")
    else:
        model = create_new_model(config, feature_count, optimizer, LOSS_FUNCTION, METRICS)

    # Handle class weights if applicable
    rtn = get_current_unit_properties(maudlin).get('class_weights_loss_function')
    if rtn:
        class_weights_loss_function_file_path = get_unit_function_path(maudlin, 'class_weights_loss_function')
        cwlfunction = load_function_from_file(class_weights_loss_function_file_path, "apply")
        model.compile(optimizer=optimizer, loss=cwlfunction, metrics=METRICS)

    model.summary()

    return model

