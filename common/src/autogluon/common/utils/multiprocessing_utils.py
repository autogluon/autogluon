import logging
import multiprocessing

logger = logging.getLogger(__name__)


# If multiprocessing_method is 'fork', initialization time scales linearly with current allocated memory, dramatically slowing down runs.
# forkserver makes this time constant
def execute_multiprocessing(workers_count, transformer, chunks, multiprocessing_method='forkserver'):
    logger.log(15, 'Execute_multiprocessing starting worker pool...')
    ctx = multiprocessing.get_context(multiprocessing_method)
    with ctx.Pool(workers_count) as pool:
        out = pool.map(transformer, chunks)
    return out
