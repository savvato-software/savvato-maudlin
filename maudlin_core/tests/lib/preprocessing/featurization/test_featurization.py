import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.lib.preprocessing.featurization.featurization import featurize, add_features, create_feature_function_map

@pytest.fixture
def groupby_config():
    return {
        'data': {
            'columns': {
                'group_by': {
                    'column': 'date',
                    'type': 'date',
                    'aggregations': [
                        {'column': 'quantity', 'agg': 'sum'},
                        {'column': 'price', 'agg': 'mean'}
                    ]
                }
            }
        }
    }

@pytest.fixture
def feature_config():
    return {
        'data': {
            'columns': {
                'csv': [
                    'age', 'job', 'marital', 'education', 'default', 'poutcome', 'contact', 'y'
                ],
                'final': [
                    'age', 'job', 'marital', 'education', 'poutcome', 'contact'
                ]
            },
            'features': [
                {
                    'name': 'frequency_encode',
                    'params': {
                        'columns': ['poutcome', 'contact']
                    }
                },
                {
                    'name': 'target_encode',
                    'params': {
                        'columns': ['job', 'marital', 'education', 'poutcome', 'contact'],
                        'smoothing': 12.0,
                        'target_column': 'y'
                    }
                },
                {
                    'name': 'drop_column',
                    'params': {
                        'columns': ['poutcome', 'contact']
                    }
                }
            ]
        }
    }

@pytest.fixture
def groupby_data():
    return pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'quantity': [10, 15, 20],
        'price': [5.0, 6.0, 7.0]
    })

@pytest.fixture
def feature_data():
    return pd.DataFrame({
        'age': [25, 35, 45],
        'job': ['admin.', 'blue-collar', 'technician'],
        'marital': ['married', 'single', 'divorced'],
        'education': ['secondary', 'primary', 'tertiary'],
        'default': ['no', 'yes', 'no'],
        'poutcome': ['success', 'failure', 'unknown'],
        'contact': ['cellular', 'telephone', 'unknown'],
        'y': [1, 0, 1]
    })

@patch('src.lib.preprocessing.featurization.featurization.load_function_from_file')
def test_create_feature_function_map(mock_load_function, feature_config):
    mock_function = MagicMock(return_value=lambda x, **params: x)
    mock_load_function.return_value = mock_function

    feature_map = create_feature_function_map(feature_config)

    assert 'frequency_encode' in feature_map
    assert 'target_encode' in feature_map
    assert 'drop_column' in feature_map
    assert callable(feature_map['frequency_encode'])

    mock_load_function.assert_called()

def test_add_features(feature_config, feature_data):
    mock_function = MagicMock(side_effect=lambda df, **params: df.assign(mock_feature=[1, 2, 3]))
    feature_function_map = {'frequency_encode': mock_function}

    result = add_features(feature_config['data']['features'], feature_function_map, feature_data)

    assert 'mock_feature' in result.columns
    assert list(result['mock_feature']) == [1, 2, 3]

def test_featurize(feature_config, feature_data):
    with patch('src.lib.preprocessing.featurization.featurization.create_feature_function_map') as mock_feature_map:
        mock_function = MagicMock(side_effect=lambda df, **params: df.assign(mock_feature=[1, 2, 3]))
        mock_feature_map.return_value = {'frequency_encode': mock_function}

        result = featurize(feature_config, feature_data)

        assert 'mock_feature' in result.columns
        assert list(result['mock_feature']) == [1, 2, 3]

def test_featurize_groupby(groupby_config, groupby_data):
    groupby_data['date'] = pd.to_datetime(groupby_data['date'])

    mock_function = MagicMock(return_value=groupby_data)
    feature_function_map = {'frequency_encode': mock_function}

    grouped_result = featurize(groupby_config, groupby_data)

    assert 'quantity' in grouped_result.columns
    assert grouped_result['quantity'].sum() == 45
    assert round(grouped_result['price'].mean(), 2) == 6.0
