import pytest
import pandas as pd


@pytest.fixture
def df_X_train():
    return pd.DataFrame({'Col_one': [1, 2, 3, 4], 'Col_two': [5,6,7,8]})

@pytest.fixture
def df_X_valid():
    return pd.DataFrame({'Col_one': [1,4], 'Col_two': [5,7]})

@pytest.fixture
def df_y_valid():
    return pd.DataFrame({'Col_target': [6,8]})

@pytest.fixture
def df_y_train():
    return pd.DataFrame({'Col_target': [5, 6, 8, 11]})

@pytest.fixture
def df_model():
    return pd.DataFrame({'Col_target': [5, 6, 8, 11]})
