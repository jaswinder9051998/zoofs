# Pytest code to test DragonFlySelectionCV class
# I will use English comments as requested.

import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import make_regression

from zoofs.feature_selection import DragonFlySelectionCV

# Sample dataset
X, y = make_regression(n_samples=50, n_features=20, noise=0.1, random_state=42)

# Test case parameters
params = [
    (RandomForestRegressor(), False),
    (PLSRegression(), True),
]

@pytest.mark.parametrize("estimator, auto_n_components", params)
def test_dragonfly_fit_transform(estimator, auto_n_components):
    """
    Test if DragonFlySelectionCV can fit and transform without any errors.
    
    Parameters
    ----------
    estimator : object
        The base estimator to use for feature selection.
    auto_n_components : bool
        Whether or not to automatically adjust the number of components for PLSRegression.
    """
    # Initialize DragonFlySelectionCV with the given parameters
    dragonfly = DragonFlySelectionCV(
        estimator=estimator,
        auto_n_components=auto_n_components,
        n_population=15,
        n_iteration=15,
        n_jobs=-1
    )

    # Test if the 'fit' method works without any errors
    dragonfly.fit(X, y)
    # Test if the 'transform' method works without any errors
    X_transformed = dragonfly.transform(X)
    # Check if the transformed array has fewer or equal features than the original array
    assert X_transformed.shape[1] <= X.shape[1]
