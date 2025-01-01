
import pytest
import pandas as pd
from src.lib.preprocessing.featurization.featurizers.target_encode import apply

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'category_1': ['A', 'B', 'A', 'B', 'A', 'C'],
        'category_2': ['X', 'Y', 'X', 'Y', 'Z', 'Z'],
        'y': [1, 0, 1, 0, 1, 0]
    })

def test_basic_functionality(sample_data):
    result = apply(sample_data, columns=['category_1', 'category_2'], target_column='y', smoothing=0.0)
    assert 'category_1' in result.columns
    assert 'category_2' in result.columns
    assert result['category_1'].iloc[0] == 1.0
    assert result['category_1'].iloc[1] == 0.0
    assert result['category_1'].iloc[5] == 0.0

def test_smoothing(sample_data):
    smoothed_result = apply(sample_data, columns=['category_1'], target_column='y', smoothing=1.0)
    assert smoothed_result['category_1'].iloc[0] > 0.5

def test_missing_values(sample_data):
    sample_data_with_missing = sample_data.copy()
    sample_data_with_missing.loc[2, 'category_1'] = None
    result_with_missing = apply(sample_data_with_missing, columns=['category_1'], target_column='y', smoothing=0.0)
    assert result_with_missing['category_1'].isnull().sum() == 0

def test_empty_dataframe():
    empty_data = pd.DataFrame(columns=['category_1', 'y'])
    result_empty = apply(empty_data, columns=['category_1'], target_column='y', smoothing=0.0)
    assert result_empty.empty

def test_no_target_values():
    no_target_data = pd.DataFrame({'category_1': ['A', 'B'], 'y': [None, None]})
    with pytest.raises(ZeroDivisionError):
        apply(no_target_data, columns=['category_1'], target_column='y', smoothing=0.0)

