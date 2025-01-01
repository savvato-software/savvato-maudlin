from .load_csv_data_get_inputs_and_targets import load_csv_data_get_inputs_and_targets

def load_for_prediction(config, output_dir):
    
    return load_csv_data_get_inputs_and_targets(
            config,

            # path to the csv input file
            config['data']['prediction_file'],
        )

