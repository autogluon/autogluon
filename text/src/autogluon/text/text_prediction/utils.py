import numpy as np
import pandas as pd
import functools
import multiprocessing as mp


def _chunk_processor(chunk, processing_fn):
    out = []
    if isinstance(chunk, pd.DataFrame):
        for idx, row in chunk.iterrows():
            out.append(processing_fn(row))
    elif isinstance(chunk, pd.Series):
        for row in chunk:
            out.append(processing_fn(row))
    else:
        raise NotImplementedError
    return out


def parallel_transform(df, processing_fn,
                       num_process=None,
                       fallback_threshold=1000):
    """Apply the function to each row of the pandas dataframe and store the results
    in a python list.

    Parameters
    ----------
    df
        Pandas Dataframe
    processing_fn
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
    if len(df) <= fallback_threshold:
        out = []
        if isinstance(df, pd.DataFrame):
            for idx, row in df.iterrows():
                out.append(processing_fn(row))
        elif isinstance(df, pd.Series):
            for row in df:
                out.append(processing_fn(row))
        else:
            raise NotImplementedError
        return out
    else:
        chunks = np.array_split(df, num_process * 2)
        with mp.Pool(num_process) as pool:
            out_l = pool.map(functools.partial(_chunk_processor, processing_fn=processing_fn),
                             chunks)
        out = sum(out_l, [])
    return out
