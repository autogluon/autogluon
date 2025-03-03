import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import pytz

from autogluon.common.utils.utils import setup_outputdir

from ..constants import LAST_CHECKPOINT

logger = logging.getLogger(__name__)


def process_save_path(path, resume: Optional[bool] = False, raise_if_exist: Optional[bool] = True):
    """
    Convert the provided path to an absolute path and check whether it is valid.
    If a path exists, either raise error or return None.
    A None path can be identified by the `setup_outputdir` to generate a random path.

    Parameters
    ----------
    path
        A provided path.
    resume
        Whether this is a path to resume training.
    raise_if_exist
        Whether to raise error if the path exists.

    Returns
    -------
    A complete and verified path or None.
    """
    path = os.path.abspath(os.path.expanduser(path))
    if resume:
        assert os.path.isfile(os.path.join(path, LAST_CHECKPOINT)), (
            f"Trying to resume training from '{path}'. "
            f"However, it does not contain the last checkpoint file: '{LAST_CHECKPOINT}'. "
            "Are you using a correct path?"
        )
    elif os.path.isdir(path) and len(os.listdir(path)) > 0:
        if raise_if_exist:
            raise ValueError(
                f"Path {path} already exists."
                "Specify a new path to avoid accidentally overwriting a saved predictor."
            )
        else:
            logger.warning(
                "A new predictor save path is created. "
                "This is to prevent you to overwrite previous predictor saved here. "
                "You could check current save path at predictor._save_path. "
                "If you still want to use this path, set resume=True"
            )
            path = None

    return path


def setup_save_path(
    resume: Optional[bool] = None,
    old_save_path: Optional[str] = None,
    proposed_save_path: Optional[str] = None,
    warn_if_exist: Optional[bool] = True,
    raise_if_exist: Optional[bool] = False,
    fit_called: Optional[bool] = None,
):
    # TODO: remove redundant folders in DDP mode
    rank = int(os.environ.get("LOCAL_RANK", 0))
    save_path = None
    if resume:
        save_path = process_save_path(path=old_save_path, resume=True)
    elif proposed_save_path is not None:  # TODO: distinguish DDP and existed predictor
        save_path = process_save_path(path=proposed_save_path, raise_if_exist=(raise_if_exist and rank == 0))
    elif old_save_path is not None:
        if fit_called:
            save_path = process_save_path(path=old_save_path, raise_if_exist=False)
        else:
            save_path = process_save_path(path=old_save_path, raise_if_exist=(raise_if_exist and rank == 0))

    if not resume:
        save_path = setup_outputdir(
            path=save_path,
            warn_if_exist=warn_if_exist,
        )
        os.makedirs(save_path, exist_ok=True)  # setup_outputdir doesn't create dir if warn_if_exist==False

    save_path = os.path.abspath(os.path.expanduser(save_path))
    logger.debug(f"save path: {save_path}")

    return save_path


def make_exp_dir(
    root_path: str,
    job_name: Optional[str] = None,
    create: Optional[bool] = True,
):
    """
    Creates the exp dir of format e.g.,: root_path/2022_01_01/job_name_12_00_00/
    This function is to better organize the training runs. It is recommended to call this
    function and pass the returned "exp_dir" to "MultiModalPredictor.fit(save_path=exp_dir)".

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
    ct = datetime.now(tz=tz)
    date_stamp = ct.strftime("%Y_%m_%d")
    time_stamp = ct.strftime("%H_%M_%S")

    # Group logs by day first
    exp_dir = os.path.join(root_path, date_stamp)

    # Then, group by run_name and hour + min + sec to avoid duplicates
    if job_name:
        exp_dir = os.path.join(exp_dir, "_".join([job_name, time_stamp]))
    else:
        exp_dir = os.path.join(exp_dir, time_stamp)

    if create:
        os.makedirs(exp_dir, mode=0o777, exist_ok=False)

    return exp_dir
