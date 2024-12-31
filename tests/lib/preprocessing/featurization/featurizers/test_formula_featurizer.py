
import pytest
import pandas as pd
import numpy as np
from maudlin_core.lib.preprocessing.featurization.featurizers.formula_featurizer import apply

# Sample test data
def test_featurizer_apply():
    # Create a sample input DataFrame
    data = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [5, 6, 7, 8],
        'C': [9, 10, 11, 12]
    })

    # Define feature specifications
    features = [
        {'name': 'sum_AB', 'formula': 'data["A"] + data["B"]'},
        {'name': 'product_AC', 'formula': 'data["A"] * data["C"]'},
        {'name': 'is_A_even', 'formula': '(data["A"] % 2 == 0).astype(int)'}
    ]

    # Apply the featurizer
    result = apply(data, features)

    # Expected output DataFrame
    expected = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [5, 6, 7, 8],
        'C': [9, 10, 11, 12],
        'sum_AB': [6, 8, 10, 12],
        'product_AC': [9, 20, 33, 48],
        'is_A_even': [0, 1, 0, 1]
    })

    print(result)

    # Validate output
    pd.testing.assert_frame_equal(result, expected)

# Test invalid feature definitions
def test_invalid_feature_definition():
    data = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })

    # Missing 'name'
    features = [{'formula': 'data["A"] + data["B"]'}]
    with pytest.raises(ValueError, match="Invalid feature definition"):
        apply(data, features)

    # Missing 'formula'
    features = [{'name': 'sum_AB'}]
    with pytest.raises(ValueError, match="Invalid feature definition"):
        apply(data, features)

# Test formula evaluation error
def test_formula_evaluation_error():
    data = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })

    # Invalid formula
    features = [{'name': 'invalid_formula', 'formula': 'data["A"] / XYZ'}]
    with pytest.raises(ValueError, match="Error processing formula"):
        apply(data, features)

