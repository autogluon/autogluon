import multiprocessing
import pandas as pd
import numpy as np


def dataframe_transform_parallel(
        df, transformer
                   ):
    cpu_count = multiprocessing.cpu_count()
    workers_count = int(round(cpu_count))

    print('running pool with', workers_count, 'workers')

    df_chunks = np.array_split(df, workers_count)

    df_list = execute_multiprocessing(workers_count=workers_count, transformer=transformer, chunks=df_chunks)

    df_combined = pd.concat(df_list, axis=0, ignore_index=True)

    return df_combined


# If multiprocessing_method is 'fork', initialization time scales linearly with current allocated memory, dramatically slowing down runs. forkserver makes this time constant
def execute_multiprocessing(workers_count, transformer, chunks, multiprocessing_method='forkserver'):
    print('starting pool...')
    ctx = multiprocessing.get_context(multiprocessing_method)
    with ctx.Pool(workers_count) as pool:
        out = pool.map(transformer, chunks)
    return out
