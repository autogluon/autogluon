def get_numpy_seed(seed: int | None) -> int | None:
    """
    Convert any integer to a valid seed for the legacy NumPy random API (RandomState).
    This ensures compatibility with NumPy 2.x on Windows while preserving as much entropy as possible.

    Parameters
    ----------
    seed : int | None
        The seed to be converted.

    Returns
    -------
    int | None
        A 32-bit integer seed compatible with NumPy's legacy RandomState.
    """
    if seed is None:
        return None
    seed = int(seed)
    if seed <= 0xFFFFFFFF:
        return seed
    # XOR-fold to mix upper 32 bits with lower 32 bits, preserving more entropy than simple truncation or modulo.
    return (seed ^ (seed >> 32)) & 0xFFFFFFFF
