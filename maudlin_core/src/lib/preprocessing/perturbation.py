import numpy as np

def apply_perturbations(config, df):
    """
    Apply configured perturbations to an entire dataframe.
    Args:
        config (dict): Configuration for perturbation.
        df (pd.DataFrame): The DataFrame containing the data.
    Returns:
        pd.DataFrame: A new DataFrame with perturbations applied.
    """
    if config['mode'] != 'prediction':
        return df

    perturbation_config = config.get('prediction', {}).get('perturbation', {})
    if not perturbation_config.get('enabled', False):
        return df

    # Create a copy to avoid unintended modifications
    perturbed_df = df.copy()

    for feature in perturbation_config.get('features', []):
        feature_name = feature['name']
        feature_type = feature.get('type')
        range_min, range_max = feature.get('range', [0, 0])
        min_value = feature.get('min', None)
        max_value = feature.get('max', None)

        if feature_name in perturbed_df.columns:
            if feature_type == 'int':
                # Generate integer perturbations for the entire column
                perturbations = np.random.randint(
                    int(range_min), int(range_max) + 1, size=len(perturbed_df)
                )
                perturbed_df[feature_name] += perturbations

            elif feature_type == 'binary':
                # Flip binary values probabilistically for the entire column
                flips = np.random.rand(len(perturbed_df)) < 0.5
                perturbed_df.loc[flips, feature_name] = 1 - perturbed_df.loc[flips, feature_name]

            elif feature_type == 'float':
                # Generate float perturbations for the entire column
                perturbations = np.random.uniform(
                    range_min, range_max, size=len(perturbed_df)
                )
                perturbed_df[feature_name] += perturbations

            else:
                raise ValueError(f"Unsupported type '{feature_type}' for feature '{feature_name}'.")

            # Apply boundaries if specified
            if min_value is not None:
                perturbed_df[feature_name] = np.maximum(perturbed_df[feature_name], min_value)
            if max_value is not None:
                perturbed_df[feature_name] = np.minimum(perturbed_df[feature_name], max_value)

            # Debug log (optional)
            print(f"Applied perturbations to '{feature_name}': type={feature_type}")

    return perturbed_df

