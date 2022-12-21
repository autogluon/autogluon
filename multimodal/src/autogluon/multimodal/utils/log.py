import datetime
import logging
import os
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import pytz

from ..constants import AUTOMM

logger = logging.getLogger(AUTOMM)


def make_exp_dir(
    root_path: str,
    job_name: str,
    create: Optional[bool] = True,
):
    """
    Creates the exp dir of format e.g.,: root_path/2022_01_01/job_name_12_00_00/
    This function is to better organize the training runs. It is recommended to call this
    function and pass the returned "exp_dir" to "AutoMMPredictor.fit(save_path=exp_dir)".

    Parameters
    ----------
    root_path
        The basic path where to create saving directories for training runs.
    job_name
        The job names to name training runs.
    create
        Whether to make the directory.

    Returns
    -------
    The formatted directory path.
    """
    tz = pytz.timezone("US/Pacific")
    ct = datetime.datetime.now(tz=tz)
    date_stamp = ct.strftime("%Y_%m_%d")
    time_stamp = ct.strftime("%H_%M_%S")

    # Group logs by day first
    exp_dir = os.path.join(root_path, date_stamp)

    # Then, group by run_name and hour + min + sec to avoid duplicates
    exp_dir = os.path.join(exp_dir, "_".join([job_name, time_stamp]))

    if create:
        os.makedirs(exp_dir, mode=0o777, exist_ok=False)

    return exp_dir


class LogFilter(logging.Filter):
    """
    Filter log messages with patterns.
    """

    def __init__(self, blacklist: Union[str, List[str]]):
        """
        Parameters
        ----------
        blacklist
            Patterns to be suppressed in logging.
        """
        super().__init__()
        if isinstance(blacklist, str):
            blacklist = [blacklist]
        self._blacklist = blacklist

    def filter(self, record):
        """
        Check whether to suppress a logging message.

        Parameters
        ----------
        record
            A logging message.

        Returns
        -------
        If True, no pattern exists in the message, hence printed out.
        If False, some pattern is in the message, hence filtered out.
        """
        matches = [pattern not in record.msg for pattern in self._blacklist]
        return all(matches)


def add_log_filter(target_logger, log_filter):
    """
    Add one log filter to the target logger.

    Parameters
    ----------
    target_logger
        Target logger
    log_filter
        Log filter
    """
    for handler in target_logger.handlers:
        handler.addFilter(log_filter)


def remove_log_filter(target_logger, log_filter):
    """
    Remove one log filter to the target logger.

    Parameters
    ----------
    target_logger
        Target logger
    log_filter
        Log filter
    """
    for handler in target_logger.handlers:
        handler.removeFilter(log_filter)


@contextmanager
def apply_log_filter(log_filter):
    """
    User contextmanager to control the scope of applying one log filter.
    Currently, it is to filter some pytorch lightning's log messages.
    But we can easily extend it to cover more loggers.

    Parameters
    ----------
    log_filter
        Log filter.
    """
    try:
        add_log_filter(logging.getLogger(), log_filter)
        add_log_filter(logging.getLogger("pytorch_lightning"), log_filter)
        yield

    finally:
        remove_log_filter(logging.getLogger(), log_filter)
        remove_log_filter(logging.getLogger("pytorch_lightning"), log_filter)
