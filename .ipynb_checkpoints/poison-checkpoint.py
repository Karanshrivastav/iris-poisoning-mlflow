# poison.py
import numpy as np
import pandas as pd

def feature_poison(X: pd.DataFrame, fraction: float, random_state: int = None) -> pd.DataFrame:
    """
    Replace features for `fraction` of rows with random numbers drawn uniformly
    across each column's observed range.
    - X: DataFrame of features (only numeric)
    - fraction: between 0 and 1 (e.g., 0.05, 0.1, 0.5)
    Returns a new DataFrame (copy).
    """
    rng = np.random.RandomState(random_state)
    Xp = X.copy().reset_index(drop=True)
    n = len(Xp)
    k = int(np.floor(n * fraction))
    if k == 0:
        return Xp
    idx = rng.choice(n, size=k, replace=False)
    for col in Xp.columns:
        col_vals = Xp[col].values
        lo, hi = col_vals.min(), col_vals.max()
        # if lo == hi, pick small spread
        if lo == hi:
            hi = lo + 1.0
        random_values = rng.uniform(low=lo, high=hi, size=k)
        col_vals[idx] = random_values
        Xp[col] = col_vals
    return Xp

def label_flip_poison(y: pd.Series, fraction: float, random_state: int = None) -> pd.Series:
    """
    Optional: flip labels for `fraction` of rows to a random other label.
    (Not used by default in examples; included as utility.)
    """
    rng = np.random.RandomState(random_state)
    ypo = y.copy().reset_index(drop=True)
    n = len(ypo)
    k = int(np.floor(n * fraction))
    if k == 0:
        return ypo
    idx = rng.choice(n, size=k, replace=False)
    classes = np.unique(ypo)
    for i in idx:
        other = rng.choice(classes[classes != ypo.iloc[i]])
        ypo.iloc[i] = other
    return ypo
