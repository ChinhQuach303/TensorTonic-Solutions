import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    """
    Standardize X: (X - mean)/std. If 2D and axis=0, per column.
    Return np.ndarray (float).
    """
    # Write code here
    if X.ndim ==1:
        X = X.reshape(1, -1)
    mean_X = np.mean(X, axis = axis, keepdims = True)
    std_X = np.std(X, axis = axis, keepdims = True)
    return (X - mean_X) / (std_X + eps)
    