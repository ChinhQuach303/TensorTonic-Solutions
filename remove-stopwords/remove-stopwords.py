import numpy as np
def remove_stopwords(tokens, stopwords):
    """
    Returns: list[str] - tokens with stopwords removed (preserve order)
    """
    # Your code here
    tokens = np.asarray(tokens)
    
    mask = np.isin(tokens, stopwords)
    return tokens[~mask].tolist()