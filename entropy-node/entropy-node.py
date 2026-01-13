import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    unique, counts = np.unique(y, return_counts = True)
    p = counts/counts.sum()
    return -np.sum(p*np.log2(p))