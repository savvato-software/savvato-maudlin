import pytest
import pandas as pd
import numpy as np
from maudlin_core.lib.preprocessing.featurization.featurizers.winsorize import apply

@pytest.fixture
def sample_data():
    # Create sample data
    return pd.DataFrame({
        'col1': [1, 2, 3, 4, 5, 100],
        'col2': [-10, 0, 10, 20, 30, 1000]
    })

# Test clipping functionality
def test_apply_clipping(sample_data):
    columns = ['col1', 'col2']
    result = apply(sample_data, columns, lower_percentile=0.01, upper_percentile=0.99)

    # Expected quantile-based clipping
    expected_col1 = np.clip(sample_data['col1'], 1.05, 95.25)
    expected_col2 = np.clip(sample_data['col2'], -9.5, 951.5)

    # Assert equality for each column
    pd.testing.assert_series_equal(result['col1'], expected_col1)
    pd.testing.assert_series_equal(result['col2'], expected_col2)

# Test no change when percentiles cover full range
def test_apply_no_clipping(sample_data):
    columns = ['col1', 'col2']
    result = apply(sample_data, columns, lower_percentile=0.0, upper_percentile=1.0)

    # Should be identical to input
    pd.testing.assert_frame_equal(result, sample_data)

# Test edge case with empty data
def test_apply_empty_dataframe():
    empty_data = pd.DataFrame(columns=['col1', 'col2'])
    result = apply(empty_data, ['col1', 'col2'])

    # Should return an empty dataframe with the same columns
    assert result.empty
    assert list(result.columns) == ['col1', 'col2']

