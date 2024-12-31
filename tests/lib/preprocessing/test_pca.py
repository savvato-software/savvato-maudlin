import pytest
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from maudlin_core.lib.preprocessing.pca import apply_pca_if_enabled, log_pca_statistics


@pytest.fixture
def sample_data():
    X_train = pd.DataFrame(np.random.rand(100, 10), columns=[f'col_{i}' for i in range(10)])
    X_test = pd.DataFrame(np.random.rand(50, 10), columns=[f'col_{i}' for i in range(10)])
    X_val = pd.DataFrame(np.random.rand(30, 10), columns=[f'col_{i}' for i in range(10)])
    columns = [f'col_{i}' for i in range(10)]
    return X_train, X_test, X_val, columns


@pytest.fixture
def pca_config_enabled():
    return {
        'pca': {
            'enabled': True,
            'params': {'n_components': 5},
            'console_logging': False
        }
    }


@pytest.fixture
def pca_config_disabled():
    return {
        'pca': {
            'enabled': False,
            'params': {'n_components': 5},
            'console_logging': False
        }
    }


def test_apply_pca_enabled(sample_data, pca_config_enabled):
    X_train, X_test, X_val, columns = sample_data
    X_train_pca, X_test_pca, X_val_pca, pca = apply_pca_if_enabled(pca_config_enabled, X_train, X_test, X_val, columns)

    assert X_train_pca is not None
    assert X_test_pca is not None
    assert X_val_pca is not None
    assert pca is not None
    assert X_train_pca.shape[1] == pca_config_enabled['pca']['params']['n_components']
    assert X_test_pca.shape[1] == pca_config_enabled['pca']['params']['n_components']
    assert X_val_pca.shape[1] == pca_config_enabled['pca']['params']['n_components']


def test_apply_pca_disabled(sample_data, pca_config_disabled):
    X_train, X_test, X_val, columns = sample_data
    X_train_pca, X_test_pca, X_val_pca, pca = apply_pca_if_enabled(pca_config_disabled, X_train, X_test, X_val, columns)

    assert X_train_pca is None
    assert X_test_pca is None
    assert X_val_pca is None
    assert pca is None


def test_apply_pca_no_test_val(sample_data, pca_config_enabled):
    X_train, _, _, columns = sample_data
    X_train_pca, X_test_pca, X_val_pca, pca = apply_pca_if_enabled(pca_config_enabled, X_train, None, None, columns)

    assert X_train_pca is not None
    assert X_test_pca is None
    assert X_val_pca is None
    assert pca is not None


def test_log_pca_statistics(sample_data):
    X_train, _, _, columns = sample_data
    pca = PCA(n_components=5)
    pca.fit(X_train)

    try:
        log_pca_statistics(pca, columns)
    except Exception as e:
        pytest.fail(f"log_pca_statistics raised an exception: {e}")

