import numpy as np

def streaming_minmax_init(D):
    """
    Initialize state dict with running min/max.
    """
    return {
        'min': np.full(D, np.inf),
        'max': np.full(D, -np.inf)
    }

def streaming_minmax_update(state, X_batch, eps=1e-8):
    """
    Update state's min/max with X_batch, return normalized batch.
    """
    X_batch = np.asarray(X_batch)

    if X_batch.ndim == 1:
        X_batch = X_batch.reshape(1, -1)

    # Update running statistics
    state['min'] = np.minimum(state['min'], X_batch.min(axis=0))
    state['max'] = np.maximum(state['max'], X_batch.max(axis=0))

    # Normalize
    denom = state['max'] - state['min']
    denom = np.maximum(denom, eps)

    return (X_batch - state['min']) / denom
