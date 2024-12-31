import pytest
import pandas as pd
from maudlin_core.lib.preprocessing.featurization.featurizers.frequency_encode import apply

@pytest.fixture
def sample_data():
    # Sample dataset for testing
    return pd.DataFrame({
        'category': ['A', 'B', 'A', 'C', 'B', 'A'],
        'type': ['X', 'Y', 'X', 'Y', 'Y', 'X']
    })


def test_apply_frequency_encoding(sample_data):
    # Define columns to encode
    columns = ['category', 'type']

    # Apply frequency encoding
    result = apply(sample_data, columns)

    # Verify the resulting DataFrame has the expected frequency columns
    assert 'category_freq' in result.columns
    assert 'type_freq' in result.columns

    # Verify frequencies are computed correctly for 'category'
    expected_category_freq = {'A': 3/6, 'B': 2/6, 'C': 1/6}
    for index, row in result.iterrows():
        assert row['category_freq'] == pytest.approx(expected_category_freq[row['category']])

    # Verify frequencies are computed correctly for 'type'
    expected_type_freq = {'X': 3/6, 'Y': 3/6}
    for index, row in result.iterrows():
        assert row['type_freq'] == pytest.approx(expected_type_freq[row['type']])


def test_apply_with_empty_dataframe():
    # Test with an empty DataFrame
    empty_df = pd.DataFrame(columns=['category', 'type'])
    columns = ['category', 'type']

    # Apply encoding
    result = apply(empty_df, columns)

    # Verify the output is still empty
    assert result.empty
    assert 'category_freq' in result.columns
    assert 'type_freq' in result.columns


def test_apply_with_missing_column(sample_data):
    # Test when one column is missing
    columns = ['category', 'non_existent']

    # Expect KeyError due to missing column
    with pytest.raises(KeyError):
        apply(sample_data, columns)

