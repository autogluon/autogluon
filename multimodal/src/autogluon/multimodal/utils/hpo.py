import logging
import os

from ..constants import AUTOMM, RAY_TUNE_CHECKPOINT

logger = logging.getLogger(AUTOMM)


def hpo_trial(sampled_hyperparameters, predictor, checkpoint_dir=None, **_fit_args):
    from ray import tune

    _fit_args[
        "hyperparameters"
    ] = sampled_hyperparameters  # The original hyperparameters is the search space, replace it with the hyperparameters sampled
    _fit_args["save_path"] = tune.get_trial_dir()  # We want to save each trial to a separate directory
    logger.debug(f"hpo trial save_path: {_fit_args['save_path']}")
    if checkpoint_dir is not None:
        _fit_args["resume"] = True
        _fit_args["ckpt_path"] = os.path.join(checkpoint_dir, RAY_TUNE_CHECKPOINT)
    predictor._fit(**_fit_args)
