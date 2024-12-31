import pytest
import pandas as pd
from maudlin_core.lib.preprocessing.featurization.featurizers.moving_average import apply

# Sample test data
data = pd.DataFrame({
    'value': [10, 20, 30, 40, 50, 60]
})

@pytest.mark.parametrize("periods, data_field_name, include_differences, expected_columns", [
    # Test case 1: Single period, no differences
    ([3], 'value', False, ['value', 'MA_3']),

    # Test case 2: Multiple periods, no differences
    ([2, 3], 'value', False, ['value', 'MA_2', 'MA_3']),

    # Test case 3: Multiple periods, with differences
    ([2, 3], 'value', True, ['value', 'MA_2', 'MA_3', 'MA_diff_2_3']),
])
def test_apply(periods, data_field_name, include_differences, expected_columns):
    # Apply featurizer
    result = apply(data.copy(), periods, data_field_name, include_differences)

    # Check that all expected columns are present
    assert list(result.columns) == expected_columns

    # Check that the moving average values are calculated correctly
    if 'MA_3' in result.columns:
        assert result['MA_3'].iloc[2] == 20  # Average of [10, 20, 30]
    if 'MA_2' in result.columns:
        assert result['MA_2'].iloc[1] == 15  # Average of [10, 20]

    # Check differences if enabled
    if 'MA_diff_2_3' in result.columns:
        assert result['MA_diff_2_3'].iloc[2] == pytest.approx(15 - 20)

