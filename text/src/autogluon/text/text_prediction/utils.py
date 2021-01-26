import numpy as np
from autogluon_contrib_nlp.utils.misc import num_mp_workers


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
        List of samples
    """
    if num_process is None:
        num_process = num_mp_workers()
    if len(df) <= fallback_threshold:
        out = []
        for idx, row in df.iterrows():
            out.append(processing_fn(row))
        return out
    else:
        chunks = np.array_split(df, num_process * 8)
        with mp.Pool(num_process) as pool:
            out_l = pool.map(functools.partial(_chunk_processor, processing_fn=processing_fn),
                             chunks)
        out = sum(out_l, [])
    return out
