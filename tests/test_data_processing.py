import pytest # type: ignore
import sys
import os
import pandas as pd


# Add project root and src folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from train import preprocess

def test_preprocess_features():
    # Create sample dataframe mimicking your real data structure
    data = {
        'Feature1': [1, 2, 3],
        'Feature2': [4, 5, 6],
        'is_high_risk': [0, 1, 0],
        'CustomerId': [10, 11, 12]
    }
    df = pd.DataFrame(data)

    # Run the function
    X, y = preprocess(df)

    # Assert the target column is removed from features
    assert 'is_high_risk' not in X.columns, "Target column should not be in features"

    # Assert CustomerId remains in original df but not in features
    assert 'CustomerId' in df.columns, "CustomerId should be present in original DataFrame"
    assert 'CustomerId' in df.columns, "CustomerId should be present in original DataFrame"
    assert 'CustomerId' not in X.columns, "CustomerId should not be in features"

    # Assert y is equal to the target column values
    assert y.tolist() == [0, 1, 0], "Target values do not match expected"

    # Assert feature columns are as expected
    expected_features = {'Feature1', 'Feature2'}
    assert set(X.columns) == expected_features, f"Features columns mismatch. Expected {expected_features}, got {set(X.columns)}"
