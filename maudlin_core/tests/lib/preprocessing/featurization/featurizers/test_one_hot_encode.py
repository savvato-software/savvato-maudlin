
import pytest
import pandas as pd
from src.lib.preprocessing.featurization.featurizers.one_hot_encode import apply

# Test valid input
def test_apply_valid_input():
    data = pd.DataFrame({
        'color': ['red', 'blue', 'green'],
        'size': ['S', 'M', 'L'],
        'price': [10, 20, 30]
    })
    columns = ['color', 'size']

    result = apply(data, columns)

    expected_columns = ['price', 'color_blue', 'color_green', 'color_red', 'size_L', 'size_M', 'size_S']
    assert list(result.columns) == expected_columns
    assert result.shape == (3, 7)
    assert result['color_red'].tolist() == [1.0, 0.0, 0.0]


# Test empty DataFrame
def test_apply_empty_dataframe():
    data = pd.DataFrame()
    columns = ['category']

    with pytest.raises(ValueError, match="Input data must be a pandas DataFrame"):
        apply([], columns)


# Test invalid columns parameter
def test_apply_invalid_columns():
    data = pd.DataFrame({
        'category': ['A', 'B', 'C']
    })

    with pytest.raises(ValueError, match="Columns parameter must be a list of column names"):
        apply(data, 'category')


# Test missing columns in DataFrame
def test_apply_missing_columns():
    data = pd.DataFrame({
        'category': ['A', 'B', 'C']
    })
    columns = ['nonexistent']

    # Expect a KeyError for missing columns
    with pytest.raises(KeyError):
        apply(data, columns)


# Test with no columns to encode
def test_apply_no_columns():
    data = pd.DataFrame({
        'price': [10, 20, 30]
    })
    columns = []

    result = apply(data, columns)

    # The result should be identical to input
    assert result.equals(data)

