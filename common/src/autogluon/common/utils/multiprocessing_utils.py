import logging
import multiprocessing

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def dataframe_transform_parallel(
        df, transformer
                   ):
    cpu_count = multiprocessing.cpu_count()
    workers_count = int(round(cpu_count))
    logger.log(15, 'Dataframe_transform_parallel running pool with '+str(workers_count)+' workers')
    df_chunks = np.array_split(df, workers_count)
    df_list = execute_multiprocessing(workers_count=workers_count, transformer=transformer, chunks=df_chunks)
    df_combined = pd.concat(df_list, axis=0, ignore_index=True)
    return df_combined


# If multiprocessing_method is 'fork', initialization time scales linearly with current allocated memory, dramatically slowing down runs. forkserver makes this time constant
def execute_multiprocessing(workers_count, transformer, chunks, multiprocessing_method='forkserver'):
    logger.log(15, 'Execute_multiprocessing starting worker pool...')
    ctx = multiprocessing.get_context(multiprocessing_method)
    with ctx.Pool(workers_count) as pool:
        out = pool.map(transformer, chunks)
    return out


def force_forkserver():
    """
    Forces forkserver multiprocessing mode if not set. This is needed for HPO and CUDA.
    The CUDA runtime does not support the fork start method: either the spawn or forkserver start method are required.
    forkserver is used because spawn is still affected by locking issues
    """
    if ('forkserver' in multiprocessing.get_all_start_methods()) & (not is_forkserver_enabled()):
        logger.warning('WARNING: changing multiprocessing start method to forkserver')
        multiprocessing.set_start_method('forkserver', force=True)


def is_forkserver_enabled():
    """
    Return True if current multiprocessing start method is forkserver.
    """
    return multiprocessing.get_start_method(allow_none=True) == 'forkserver'


def is_fork_enabled():
    """
    Return True if current multiprocessing start method is fork.
    """
    return multiprocessing.get_start_method(allow_none=True) == 'fork'
