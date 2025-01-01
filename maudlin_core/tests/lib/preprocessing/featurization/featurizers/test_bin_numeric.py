
import pytest
import pandas as pd
from src.lib.preprocessing.featurization.featurizers.bin_numeric import apply

# Test valid binning
def test_apply_valid_bins():
    # Sample input data
    data = pd.DataFrame({'age': [15, 25, 35, 45, 55]})
    bins = [0, 20, 40, 60]  # Bin edges
    labels = ['young', 'middle-aged', 'senior']  # Bin labels

    # Apply the function
    result = apply(data, column='age', bins=bins, labels=labels)

    # Expected output
    expected = pd.DataFrame({'age_binned': ['young', 'middle-aged', 'middle-aged', 'senior', 'senior']})
    expected['age_binned'] = pd.Categorical(expected['age_binned'], categories=labels, ordered=True)

    # Assert result matches expected output
    pd.testing.assert_frame_equal(result, expected)


# Test invalid bin-label mismatch
def test_apply_invalid_bins_labels():
    data = pd.DataFrame({'age': [15, 25, 35]})
    bins = [0, 20, 40]
    labels = ['young']  # Mismatched length

    # Assert ValueError is raised
    with pytest.raises(ValueError, match="The number of bins must be one more than the number of labels."):
        apply(data, column='age', bins=bins, labels=labels)


def test_apply_custom_column_name():
    data = pd.DataFrame({'age': [10, 20, 30, 40]})
    bins = [0, 15, 35, float('inf')]  # Expanded bins to cover 40
    labels = ['young', 'adult', 'senior']  # Added extra label
    new_column_name = 'age_group'

    # Apply function
    result = apply(data, column='age', bins=bins, labels=labels, new_column_name=new_column_name)

    # Expected output
    expected = pd.DataFrame({'age_group': ['young', 'adult', 'adult', 'senior']})
    expected['age_group'] = pd.Categorical(expected['age_group'], categories=labels, ordered=True)

    # Assert result matches expected
    pd.testing.assert_frame_equal(result, expected)


# Test empty DataFrame
def test_apply_empty_dataframe():
    data = pd.DataFrame({'age': []})  # Empty DataFrame
    bins = [0, 20, 40]
    labels = ['young', 'adult']

    result = apply(data, column='age', bins=bins, labels=labels)

    expected = pd.DataFrame({'age_binned': []})
    expected['age_binned'] = pd.Categorical(expected['age_binned'], categories=labels, ordered=True)

    pd.testing.assert_frame_equal(result, expected)


# Test dropping original column
def test_apply_column_dropped():
    data = pd.DataFrame({'age': [10, 20, 30]})
    bins = [0, 15, 35]
    labels = ['young', 'adult']

    result = apply(data, column='age', bins=bins, labels=labels)

    # Check if the original column 'age' is dropped
    assert 'age' not in result.columns
    assert 'age_binned' in result.columns

