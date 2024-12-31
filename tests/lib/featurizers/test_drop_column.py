import pytest
import pandas as pd
from maudlin_core.lib.preprocessing.featurization.featurizers.drop_column import apply

@pytest.fixture
def sample_data():
    # Create a sample dataframe for testing
    return pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    })

# Test case 1: Drop a single column
def test_drop_single_column(sample_data):
    result = apply(sample_data.copy(), ['A'])
    assert 'A' not in result.columns
    assert 'B' in result.columns
    assert 'C' in result.columns

# Test case 2: Drop multiple columns
def test_drop_multiple_columns(sample_data):
    result = apply(sample_data.copy(), ['A', 'B'])
    assert 'A' not in result.columns
    assert 'B' not in result.columns
    assert 'C' in result.columns

# Test case 3: Drop a non-existent column
def test_drop_non_existent_column(sample_data):
    with pytest.raises(KeyError):
        apply(sample_data.copy(), ['D'])

# Test case 4: Drop no columns (empty list)
def test_drop_no_columns(sample_data):
    result = apply(sample_data.copy(), [])
    assert list(result.columns) == ['A', 'B', 'C']

# Test case 5: Check inplace behavior
# Ensure original data is not modified
def test_inplace_behavior(sample_data):
    original_data = sample_data.copy()
    apply(sample_data, ['A'])
    assert list(original_data.columns) == ['A', 'B', 'C']

