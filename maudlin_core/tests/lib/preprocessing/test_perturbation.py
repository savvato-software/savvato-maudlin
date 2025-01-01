import pytest
import pandas as pd
import numpy as np
from maudlin_core.lib.preprocessing.perturbation import apply_perturbations

@pytest.fixture
def sample_data():
    # Create sample dataframe for testing
    data = {
        'age': [25, 30, 35, 40],
        'housing': [1, 0, 1, 0],
        'income': [50000.0, 60000.0, 70000.0, 80000.0]
    }
    return pd.DataFrame(data)

@pytest.fixture
def perturbation_config():
    # Example configuration for perturbations
    return {
        'mode': 'prediction',
        'prediction': {
            'perturbation': {
                'enabled': True,
                'features': [
                    {'name': 'age', 'type': 'int', 'range': [-5, 5], 'min': 20, 'max': 50},
                    {'name': 'housing', 'type': 'binary'},
                    {'name': 'income', 'type': 'float', 'range': [-1000.0, 1000.0], 'min': 40000.0, 'max': 90000.0}
                ]
            }
        }
    }

def test_apply_perturbations_int(sample_data, perturbation_config):
    # Test integer perturbations
    perturbed_df = apply_perturbations(perturbation_config, sample_data)

    assert 'age' in perturbed_df.columns
    assert all(20 <= perturbed_df['age']) and all(perturbed_df['age'] <= 50)

def test_apply_perturbations_binary(sample_data, perturbation_config):
    # Test binary perturbations
    perturbed_df = apply_perturbations(perturbation_config, sample_data)

    assert 'housing' in perturbed_df.columns
    assert set(perturbed_df['housing'].unique()).issubset({0, 1})

def test_apply_perturbations_float(sample_data, perturbation_config):
    # Test float perturbations
    perturbed_df = apply_perturbations(perturbation_config, sample_data)

    assert 'income' in perturbed_df.columns
    assert all(40000.0 <= perturbed_df['income']) and all(perturbed_df['income'] <= 90000.0)

def test_apply_perturbations_disabled(sample_data):
    # Test with perturbations disabled
    config = {'mode': 'prediction', 'prediction': {'perturbation': {'enabled': False}}}
    perturbed_df = apply_perturbations(config, sample_data)

    pd.testing.assert_frame_equal(sample_data, perturbed_df)

def test_apply_perturbations_non_prediction_mode(sample_data):
    # Test when mode is not 'prediction'
    config = {'mode': 'training'}
    perturbed_df = apply_perturbations(config, sample_data)

    pd.testing.assert_frame_equal(sample_data, perturbed_df)

def test_invalid_feature_type(sample_data):
    # Test with an invalid feature type
    config = {
        'mode': 'prediction',
        'prediction': {
            'perturbation': {
                'enabled': True,
                'features': [
                    {'name': 'age', 'type': 'unsupported', 'range': [-5, 5]}
                ]
            }
        }
    }
    with pytest.raises(ValueError, match="Unsupported type 'unsupported' for feature 'age'."):
        apply_perturbations(config, sample_data)


