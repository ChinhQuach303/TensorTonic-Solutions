import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss (vectorized).

    Parameters
    ----------
    y_true : array-like
        Ground truth values
    y_pred : array-like
        Predicted values
    delta : float
        Threshold parameter δ

    Returns
    -------
    float
        Mean Huber loss
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    error = y_true - y_pred
    abs_error = np.abs(error)

    quadratic = 0.5 * error**2
    linear = delta * (abs_error - 0.5 * delta)

    loss = np.where(abs_error <= delta, quadratic, linear)
    return np.mean(loss)
