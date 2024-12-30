import os
import yaml

from .maudlin import load_maudlin_data, get_current_unit_properties

maudlin = load_maudlin_data()

def get_current_unit_config(dir=None):
    # Define the default configuration path
    DEFAULT_CONFIG_FILE = maudlin['data-directory'] + get_current_unit_properties(maudlin)['config-path']

    config = None
    
    if dir:
        # Define the directory-specific configuration path
        custom_config_file = os.path.join(dir, "config.yaml")
    else:
        custom_config_file = ''

    # Check if the config file exists in the specified directory
    if os.path.exists(custom_config_file):
        config_file_to_load = custom_config_file
    else:
        config_file_to_load = DEFAULT_CONFIG_FILE

    # Load the configuration file
    with open(config_file_to_load, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    return config

