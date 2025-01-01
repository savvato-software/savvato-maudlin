
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from maudlin_core.lib.preprocessing.featurization.featurization import featurize, add_features, create_feature_function_map

@pytest.fixture
def sample_config():
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
                },
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
                        'columns': ['job', 'marital', 'education', 'month', 'poutcome', 'contact'],
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
def sample_data():
    return pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'quantity': [10, 15, 20],
        'price': [5.0, 6.0, 7.0]
    })

@patch('maudlin_core.lib.savvato_python_functions.load_function_from_file')
def test_create_feature_function_map(mock_load_function, sample_config):

    import inspect
    from maudlin_core.lib.savvato_python_functions import load_function_from_file
    print(f"Mocked: {inspect.getfile(load_function_from_file)}")

    # Mock the function loading process
    mock_function = MagicMock(return_value=lambda x, **params: x)
    mock_load_function.return_value = mock_function

    feature_map = create_feature_function_map(sample_config)

    # Verify the function map contains the expected feature
    assert 'frequency_encode' in feature_map
    assert 'target_encode' in feature_map
    assert 'drop_column' in feature_map
    assert callable(feature_map['frequency_encode'])

    # Ensure the function loader was called correctly
    mock_load_function.assert_called()


def test_add_features(sample_config, sample_data):
    # Mock feature function
    mock_function = MagicMock(side_effect=lambda df, **params: df.assign(mock_feature=[1, 2, 3]))
    feature_function_map = {'frequency_encode': mock_function}

    # Apply features
    result = add_features(sample_config['data']['features'], feature_function_map, sample_data)

    # Verify the new column exists
    assert 'mock_feature' in result.columns
    assert list(result['mock_feature']) == [1, 2, 3]


def test_featurize(sample_config, sample_data):
    # Mock feature map creation
    with patch('maudlin_core.lib.preprocessing.featurization.featurization.create_feature_function_map') as mock_feature_map:
        mock_function = MagicMock(side_effect=lambda df, **params: df.assign(mock_feature=[1, 2, 3]))
        mock_feature_map.return_value = {'frequency_encode': mock_function}

        # Process features
        result = featurize(sample_config, sample_data)

        # Check results
        assert 'mock_feature' in result.columns
        assert list(result['mock_feature']) == [1, 2, 3]


def test_featurize_groupby(sample_config, sample_data):
    # Update sample data to test group_by logic
    sample_data['date'] = pd.to_datetime(sample_data['date'])

    # Mock feature function
    mock_function = MagicMock(return_value=sample_data)
    feature_function_map = {'frequency_encode': mock_function}

    # Apply grouping logic
    grouped_result = featurize(sample_config, sample_data)

    # Verify aggregation logic
    assert 'quantity' in grouped_result.columns
    assert grouped_result['quantity'].sum() == 45
    assert round(grouped_result['price'].mean(), 2) == 6.0

