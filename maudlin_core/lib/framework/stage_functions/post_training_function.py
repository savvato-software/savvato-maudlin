import os

from pathlib import Path
from maudlin import load_maudlin_data, get_unit_function_path
from savvato_python_functions.savvato_python_functions import load_function_from_file
from history import update_history


def execute_posttraining_stage(config, data_dir, model, X_train, y_train, X_test, y_true):
    # Load Maudlin framework-level configuration and data
    maudlin = load_maudlin_data()

    data_dir += "/post_training"

    # Retrieve base path for functions (assumes stored in functions/)
    base_function_file_path = maudlin['output-function-directory'] + "/post_training"
    os.makedirs(data_dir, exist_ok=True)

    predictions = model.predict(X_test)
    threshold = config.get('prediction', {}).get('threshold', 0.5)
    y_preds = (predictions >= threshold).astype('int')

    # Get diagrams to generate from config
    diagrams = config['post_training'].get('diagrams', [])

    # Generate outputs dynamically
    for diagram in diagrams:
        try:
            # Dynamically load function for this diagram
            func_path = os.path.join(base_function_file_path, diagram + ".py")
            func = load_function_from_file(func_path, "generate")

            # Call the generate function with required params
            func(config, data_dir, model, X_train, y_train, X_test, y_true, y_preds)

        except Exception as e:
            print(f"Error generating diagram '{diagram}': {e}")

    print("Post-training outputs generated successfully.")

    update_history(config, Path(data_dir).parent.parent)

    post_training_function_file_path = get_unit_function_path(maudlin, 'post-training-function')
    post_training_function = load_function_from_file(post_training_function_file_path, "apply")

    return post_training_function(config, data_dir, model, X_train, y_train, X_test, y_true)

