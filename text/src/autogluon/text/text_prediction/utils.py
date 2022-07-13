import numpy as np
import logging
import os
import multiprocessing as mp
from typing import List, Optional
import inspect


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


def get_trimmed_lengths(lengths: List[int],
                        max_length: int,
                        do_merge: bool = False) -> np.ndarray:
    """Get the trimmed lengths of multiple text data. It will make sure that
    the trimmed length is smaller than or equal to the max_length

    - do_merge is True
        Make sure that sum(trimmed_lengths) <= max_length.
        The strategy is to always try to trim the longer lengths.
    - do_merge is False
        Make sure that all(trimmed_lengths <= max_length)

    Parameters
    ----------
    lengths
        The original lengths of each sample
    max_length
        When do_merge is True,
            We set the max_length constraint on the total length.
        When do_merge is False,
            We set the max_length constraint on individual sentences.
    do_merge
        Whether these sentences will be merged

    Returns
    -------
    trimmed_lengths
        The trimmed lengths of the
    """
    lengths = np.array(lengths)
    if do_merge:
        total_length = sum(lengths)
        if total_length <= max_length:
            return lengths
        trimmed_lengths = np.zeros_like(lengths)
        while sum(trimmed_lengths) != max_length:
            remainder = max_length - sum(trimmed_lengths)
            budgets = lengths - trimmed_lengths
            nonzero_idx = (budgets > 0).nonzero()[0]
            nonzero_budgets = budgets[nonzero_idx]
            if remainder < len(nonzero_idx):
                for i in range(remainder):
                    trimmed_lengths[nonzero_idx[i]] += 1
            else:
                increment = min(min(nonzero_budgets), remainder // len(nonzero_idx))
                trimmed_lengths[nonzero_idx] += increment
        return trimmed_lengths
    else:
        return np.minimum(lengths, max_length)


def logging_config(folder: Optional[str] = None,
                   name: Optional[str] = None,
                   logger: logging.Logger = logging.root,
                   level: int = logging.INFO,
                   console_level: int = logging.INFO,
                   console: bool = True,
                   overwrite_handler: bool = False) -> str:
    """Config the logging module. It will set the logger to save to the specified file path.
    Parameters
    ----------
    folder
        The folder to save the log
    name
        Name of the saved
    logger
        The logger
    level
        Logging level
    console_level
        Logging level of the console log
    console
        Whether to also log to console
    overwrite_handler
        Whether to overwrite the existing handlers in the logger
    Returns
    -------
    folder
        The folder to save the log file.
    """
    if name is None:
        name = inspect.stack()[-1][1].split('.')[0]
    if folder is None:
        folder = os.path.join(os.getcwd(), name)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    need_file_handler = True
    need_console_handler = True
    # Check all loggers.
    if overwrite_handler:
        logger.handlers = []
    else:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                need_console_handler = False
    logpath = os.path.join(folder, name + ".log")
    print("All Logs will be saved to {}".format(logpath))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if need_file_handler:
        logfile = logging.FileHandler(logpath)
        logfile.setLevel(level)
        logfile.setFormatter(formatter)
        logger.addHandler(logfile)
    if console and need_console_handler:
        # Initialze the console logging
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logger.addHandler(logconsole)
    return folder
