import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # Write code here
    a = np.asarray(a)
    b = np.asarray(b)
    # handle zeros vector
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))