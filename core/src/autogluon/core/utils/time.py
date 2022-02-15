import logging
import time
from typing import Optional

from pandas import DataFrame

logger = logging.getLogger(__name__)


def time_func(f, args: list = None, kwargs: dict = None, time_limit: float = 0.2, max_repeats: int = 10) -> float:
    """
    Returns the average time taken by `f(*args, **kwargs)`.

    Parameters
    ----------
    f : callable
        Function to be called.
        Must not alter outer context or otherwise could have negative side effects.
    args : list, default = None
        args to f
    kwargs : dict, default = None
        kwargs to f
    time_limit : float, default = 0.2
        Time limit in seconds to spend repeatedly calling `f`.
        Once time limit is exceeded, return current estimate.
    max_repeats : int, default = 10
        Maximum repeats of calling `f` to obtain a better average time estimate.

    Returns
    -------
    Average time taken in seconds by `f(*args, **kwargs)`.
    """
    if args is None:
        args = list()
    if kwargs is None:
        kwargs = dict()
    time_start = time.time()
    f(*args, **kwargs)
    time_end = time.time()
    avg_time = time_end - time_start
    if max_repeats == 1:
        return avg_time
    if time_limit is None or avg_time < (time_limit / max_repeats):
        time_start_loop = time.time()
        for i in range(max_repeats-1):
            f(*args, **kwargs)
        time_end_loop = time.time()
        total_time_loop = time_end_loop - time_start_loop
        avg_time = (avg_time + total_time_loop) / max_repeats
    return avg_time


def sample_df_for_time_func(df: DataFrame, sample_size: int, max_sample_size: Optional[int] = 10000) -> DataFrame:
    """
    Samples df for the purposes of passing as an argument to `time_func`.
    A common use-case is for batch-inference speed calculations of batch = sample_size.
    The rows of data returned should not be important beyond that each row is a valid pre-existing row in df (which could appear multiple times).

    Parameters
    ----------
    df : DataFrame
        The DataFrame to sample from.
    sample_size : int
        The sample_size to try to return such that len(X_out) == sample_size.
    max_sample_size : int, default = 10000
        The sample_size to limit the result to if sample_size>max_sample_size and max_sample_size>len(df)
        This avoids df_out taking up large amounts of memory when it shouldn't be necessary for the purposes of timing.
        For example, avoids crash if user specifies sample_size=99999999.
        If len(df)>max_sample_size and sample_size>len(df), then df is returned unaltered.
        If None, then max_sample_size == sample_size.

    Returns
    -------
    df_out : DataFrame
        Sampled DataFrame. User can get the effective sample_size via len(df_out).
    """
    len_df = len(df)
    if max_sample_size is None:
        max_sample_size = sample_size
    assert isinstance(sample_size, int), 'sample_size must be of type int'
    assert isinstance(max_sample_size, int), 'max_sample_size must be of type int'
    if sample_size < 1:
        raise AssertionError(f'sample_size must be >=1, but was {sample_size}')
    if max_sample_size < 1:
        raise AssertionError(f'max_sample_size must be >=1, but was {max_sample_size}')

    if len_df > sample_size:
        df_out = df.head(sample_size)
    elif len_df < sample_size and len_df < max_sample_size:
        sample_size = min(sample_size, max_sample_size)  # No need to sample more than 10,000 generally
        df_out = df.sample(sample_size, replace=True, random_state=0, ignore_index=True)
    else:
        # Not enough rows of data to satisfy batch size, instead use all available
        df_out = df
    return df_out
