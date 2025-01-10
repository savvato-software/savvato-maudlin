import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from maudlin_core.src.lib.preprocessing.pca import apply_pca_if_enabled
from imblearn.over_sampling import SMOTE, ADASYN
from maudlin_core.src.lib.framework.maudlin import load_maudlin_data, get_unit_function_path
from maudlin_core.src.lib.savvato_python_functions import load_function_from_file

# Load Maudlin configuration and data
maudlin = load_maudlin_data()

def execute_pretraining_stage(config, data_dir, X_train, y_train, X_test, y_test, X_val, y_val, columns):
    if data_dir:
        data_dir += "/pre_training"
        os.makedirs(data_dir, exist_ok=True)

    oversampling_config = config['pre_training'].get('oversampling', {})
    run_pca_before_oversampling = oversampling_config.get('run_pca_before', False)

    if run_pca_before_oversampling:
        X_train_pca, X_test_pca, X_val_pca, pca = apply_pca_if_enabled(config, X_train, X_test, X_val, columns)
        X_train_resampled, y_train_resampled = apply_oversampling_if_enabled(config, X_train_pca if X_train_pca is not None else X_train, y_train, pca)
    else:
        X_train_resampled, y_train_resampled = apply_oversampling_if_enabled(config, X_train, y_train, None)
        X_train_pca, X_test_pca, X_val_pca, pca = apply_pca_if_enabled(config, X_train_resampled if X_train_resampled is not None else X_train, X_test, X_val, columns)

    generate_visualizations(config, data_dir, X_train_resampled if X_train_resampled is not None else X_train, X_train_resampled, y_train_resampled, pca, columns)

    pre_training_function_file_path = get_unit_function_path(maudlin, 'pre-training-function')
    pre_training_function = load_function_from_file(pre_training_function_file_path, "apply")

    X_train_final, y_train_final, X_test_final, y_test_final, X_val_final, y_val_final = pre_training_function(
        config,
        data_dir,
        X_train_pca if X_train_pca is not None else X_train_resampled if X_train_resampled is not None else X_train,
        y_train_resampled if y_train_resampled is not None else y_train,
        X_test_pca if X_test_pca is not None else X_test,
        y_test,
        X_val_pca if X_val_pca is not None else X_val,
        y_val
    )

    # Filter selected features based on config
    if config['mode'] == 'training':
        selected_indices = config['pre_training'].get('optimized_feature_indices', [])

        def filter_features_ndarray(X, selected_indices):
            if len(selected_indices) > 0:
                return X[:, selected_indices]  # Select only the specified columns
            return X  # If no selection, return as is

        # Apply filtering to final datasets
        X_train_final = filter_features_ndarray(X_train_final, selected_indices)
        X_test_final = filter_features_ndarray(X_test_final, selected_indices)
        X_val_final = filter_features_ndarray(X_val_final, selected_indices)

        print(f"\nSelected features: {selected_indices}\n")

    return X_train_final, y_train_final, X_test_final, y_test_final, X_val_final, y_val_final



def apply_oversampling_if_enabled(config, X_train, y_train, pca):
    # Extract oversampling configuration
    oversampling_config = config['pre_training'].get('oversampling', {})
    if not oversampling_config.get('enabled', False):
        return None, None

    # Select oversampling method and extract parameters
    method = oversampling_config.get('method', 'smote').lower()
    sampling_strategy = oversampling_config.get('sampling_strategy', 1.0)  # Balance by default
    k_neighbors = oversampling_config.get('k_neighbors', 5)  # Default 5 neighbors
 #   m_neighbors = oversampling_config.get('m_neighbors', 10)  # ADASYN-specific default
#    threshold_cleaning = oversampling_config.get('threshold_cleaning', 0.5)  # ADASYN-specific
    random_state = oversampling_config.get('random_state', 42)  # Reproducibility
    n_jobs = oversampling_config.get('n_jobs', None)  # Parallelization

    # Initialize oversampler based on method
    if method == 'smote':
        oversampler = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=random_state,
            n_jobs=n_jobs
        )
    elif method == 'adasyn':
        oversampler = ADASYN(
            sampling_strategy=sampling_strategy,
            n_neighbors=k_neighbors,  # Uses k_neighbors instead of m_neighbors
            random_state=random_state,
            n_jobs=n_jobs,
#            m_neighbors=m_neighbors,
#            threshold_cleaning=threshold_cleaning
        )
    else:
        raise ValueError(f"Unsupported oversampling method: {method}")

    print(f"Applying {method.upper()} to {'PCA-transformed data' if pca else 'original data'}...")

    # Fit and resample data
    X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

    # Log updated class distribution
    if oversampling_config.get('console_logging', False):
        unique, counts = np.unique(y_train_resampled, return_counts=True)
        print(f"Class distribution after {method.upper()}: {dict(zip(unique, counts))}")

    return X_train_resampled, y_train_resampled


def generate_visualizations(config, data_dir, X_train, X_resampled, y_resampled, pca, columns):

    if not data_dir:
        return

    # Get diagrams to generate from config
    diagrams = config['pre_training'].get('diagrams', [])
    base_function_file_path = maudlin['output-function-directory'] + "/pre_training"
    os.makedirs(data_dir, exist_ok=True)

    # Generate outputs dynamically
    for diagram in diagrams:
        try:
            # Dynamically load function for this diagram
            func_path = os.path.join(base_function_file_path, diagram + ".py")
            func = load_function_from_file(func_path, "generate")

            # Call the generate function with required params
            func(config, data_dir, X_train, X_resampled, y_resampled, columns, pca)

        except Exception as e:
            print(f"Error generating diagram '{diagram}': {e}")

    print("Post-prediction outputs generated successfully.")

   
