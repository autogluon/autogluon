import logging
import multiprocessing

logger = logging.getLogger(__name__)


# If multiprocessing_method is 'fork', initialization time scales linearly with current allocated memory, dramatically slowing down runs.
# forkserver makes this time constant
def execute_multiprocessing(workers_count, transformer, chunks, multiprocessing_method="forkserver"):
    logger.log(15, "Execute_multiprocessing starting worker pool...")
    if multiprocessing_method in ("fork", "forkserver") and (
        multiprocessing_method not in multiprocessing.get_all_start_methods()
    ):
        # 'fork' and 'forkserver' are POSIX-only; Windows offers only 'spawn'. Any other name goes to
        # multiprocessing.get_context() unchanged, which raises ValueError if it is not a start method.
        logger.log(
            15,
            f"multiprocessing_method='{multiprocessing_method}' is unavailable on this platform "
            f"(available: {multiprocessing.get_all_start_methods()}); falling back to 'spawn'.",
        )
        multiprocessing_method = "spawn"
    ctx = multiprocessing.get_context(multiprocessing_method)
    with ctx.Pool(workers_count) as pool:
        out = pool.map(transformer, chunks)
    return out
