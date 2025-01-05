import os

from maudlin_core.src.lib.framework.maudlin import load_maudlin_data, get_unit_function_path

from maudlin_core.src.lib.preprocessing.pca import apply_pca_if_enabled
from maudlin_core.src.lib.savvato_python_functions import load_function_from_file

# Load Maudlin configuration and data
maudlin = load_maudlin_data()

def execute_preprediction_stage(config, data_dir, model, X, y_true, feature_count, columns):
    print("Pre-training mode begin...")

    data_dir += "/pre_prediction"

    # Retrieve base path for functions (assumes stored in functions/)
    base_function_file_path = maudlin['output-function-directory'] + "/pre_prediction"
    os.makedirs(data_dir, exist_ok=True)

    # Get diagrams to generate from config
    diagrams = config['pre_prediction'].get('diagrams', [])

    if diagrams:
        # Generate outputs dynamically
        for diagram in diagrams:
            try:
                # Dynamically load function for this diagram
                func_path = os.path.join(base_function_file_path, diagram + ".py")
                func = load_function_from_file(func_path, "generate")

                # Call the generate function with required params
                func(config, data_dir, X, y_true)

            except Exception as e:
                print(f"Error generating diagram '{diagram}': {e}")

        print("Pre-prediction outputs generated successfully.")

    pre_prediction_function_file_path = get_unit_function_path(maudlin, 'pre-prediction-function')
    pre_prediction_function = load_function_from_file(pre_prediction_function_file_path, "apply")

    pre_prediction_function(config, data_dir, model)

    return apply_pca_if_enabled(config, X, None, None, columns)

