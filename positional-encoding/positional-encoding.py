import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    pe = np.zeros((seq_len, d_model), dtype=float)

    position = np.arange(seq_len)[:, None]              # (T, 1)
    div_term = base ** (2 * (np.arange(d_model) // 2) / d_model)

    pe[:, 0::2] = np.sin(position / div_term[0::2])
    pe[:, 1::2] = np.cos(position / div_term[1::2])

    return pe


def add_positional_encoding(x, base=10000.0):
    """
    Add PE to input x of shape (B, T, d_model); return same shape.
    """
    pe = positional_encoding(x.shape[1], x.shape[2], base)
    return x + pe
