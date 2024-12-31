import pytest
import pandas as pd
import numpy as np
from maudlin_core.lib.preprocessing.featurization.featurizers.min_max_scale import apply

# Test cases for the Min-Max scaling featurizer
def test_apply_valid_input():
    # Test with valid input
    data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50]
    })
    columns = ['A', 'B']

    result = apply(data.copy(), columns)

    assert result['A'].min() == 0.0
    assert result['A'].max() == 1.0
    assert result['B'].min() == 0.0
    assert result['B'].max() == 1.0


def test_apply_invalid_data_type():
    # Test with invalid data type for input
    with pytest.raises(ValueError, match="Input data must be a pandas DataFrame."):
        apply([1, 2, 3], ['A'])


def test_apply_invalid_columns_type():
    # Test with invalid columns type
    data = pd.DataFrame({'A': [1, 2, 3]})
    with pytest.raises(ValueError, match="Columns parameter must be a list of column names."):
        apply(data, 'A')


def test_apply_missing_column():
    # Test with a missing column in the DataFrame
    data = pd.DataFrame({'A': [1, 2, 3]})
    with pytest.raises(ValueError, match="Column 'B' not found in DataFrame."):
        apply(data, ['B'])


def test_apply_constant_column():
    # Test with a constant column (min == max)
    data = pd.DataFrame({'A': [5, 5, 5]})
    columns = ['A']
    result = apply(data.copy(), columns)

    assert (result['A'] == 0.0).all()


def test_apply_empty_dataframe():
    # Test with an empty DataFrame
    data = pd.DataFrame()
    columns = []
    result = apply(data.copy(), columns)
    assert result.empty


def test_apply_single_value_column():
    # Test with a column having a single unique value
    data = pd.DataFrame({'A': [3] * 5})
    columns = ['A']
    result = apply(data.copy(), columns)
    assert (result['A'] == 0.0).all()

