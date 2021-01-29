import numpy as np
import pandas as pd
import functools
import multiprocessing as mp


def parallel_transform(df, chunk_processor,
                       num_process=None,
                       fallback_threshold=1000):
    """Apply the function to each row of the pandas dataframe and store the results
    in a python list.

    Parameters
    ----------
    df
        Pandas Dataframe
    chunk_processor
        The processing function
    num_process
        If not set. We use the default value
    fallback_threshold
        If the number of samples in df is smaller than fallback_threshold.
        Directly transform the data without multiprocessing

    Returns
    -------
    out
        The output
    """
    if num_process is None:
        num_process = mp.cpu_count()
    num_process = max(num_process, 1)
    if len(df) <= fallback_threshold or num_process == 1:
        return chunk_processor(df)
    else:
        chunks = np.array_split(df, num_process * 2)
        with mp.Pool(num_process) as pool:
            out_l = pool.map(chunk_processor, chunks)
        out = sum(out_l, [])
    return out
