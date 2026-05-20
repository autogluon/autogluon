def get_numpy_seed(seed: int | None) -> int | None:
    """
    Convert any integer (including negative or 64-bit) to a valid 32-bit unsigned seed
    for the legacy NumPy random API (RandomState).
    This ensures compatibility with NumPy 2.x on Windows while preserving as much entropy as possible.

    Parameters
    ----------
    seed : int | None
        The seed to be converted.

    Returns
    -------
    int | None
        A non-negative 32-bit integer seed compatible with NumPy's legacy RandomState.
    """
    if seed is None:
        return None
    seed = int(seed)

    # If the seed is already within the valid unsigned 32-bit range, we use it directly.
    if 0 <= seed <= 0xFFFFFFFF:
        return seed

    # If the seed is negative or larger than 32-bit, we use a bit-mixing (XOR-fold) technique.
    # The & 0xFFFFFFFF mask guarantees a positive 32-bit integer in Python for any input.
    return (seed ^ (seed >> 32)) & 0xFFFFFFFF
