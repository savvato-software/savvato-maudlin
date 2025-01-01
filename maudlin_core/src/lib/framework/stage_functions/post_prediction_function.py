import os
from ..maudlin import load_maudlin_data, get_unit_function_path
from ...savvato_python_functions import load_function_from_file

def execute_postprediction_stage(config, data_dir, model, predictions, y_true):

    maudlin = load_maudlin_data()

    # Retrieve base path for functions (assumes stored in functions/)
    base_function_file_path = maudlin['output-function-directory'] + "/post_prediction"

#    predictions = model.predict(X_test)
    threshold = config.get('prediction', {}).get('threshold', 0.5)
    y_preds = (predictions >= threshold).astype('int')

    # Get diagrams to generate from config
    diagrams = config['post_prediction'].get('diagrams', [])

    # Generate outputs dynamically
    for diagram in diagrams:
        try:
            # Dynamically load function for this diagram
            func_path = os.path.join(base_function_file_path, diagram + ".py")
            func = load_function_from_file(func_path, "generate")

            # Call the generate function with required params
            func(y_preds, data_dir, y_true)

        except Exception as e:
            print(f"Error generating diagram '{diagram}': {e}")

    print("Post-prediction outputs generated successfully.")

    post_prediction_function_file_path = get_unit_function_path(maudlin, 'post-prediction-function')
    post_prediction_function = load_function_from_file(post_prediction_function_file_path, "apply")

    return post_prediction_function(config, model, predictions, y_true, data_dir + '/post_prediction')

