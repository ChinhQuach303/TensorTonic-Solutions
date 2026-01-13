import numpy as np

def gini(y):
    if len(y) == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1.0 - np.sum(p ** 2)


def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    N_left = len(y_left)
    N_right = len(y_right)
    N = N_left + N_right

    if N == 0:
        return 0.0

    return (
        (N_left / N) * gini(y_left)
        + (N_right / N) * gini(y_right)
    )
