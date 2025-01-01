from ..framework.maudlin_unit_config import get_current_unit_config
from .load_csv_data_get_inputs_and_targets import load_csv_data_get_inputs_and_targets

def load_for_training(config, output_dir, generate_targets=True):
    
    return load_csv_data_get_inputs_and_targets(
            config,
            
            # path to the csv input file
            config['data']['training_file'],

            generate_targets
        )

