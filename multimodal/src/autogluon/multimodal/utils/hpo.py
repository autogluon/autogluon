import logging
import os
import shutil

import lightning.pytorch as pl
import yaml

from autogluon.common.utils.context import set_torch_num_threads

from ..constants import BEST_K_MODELS_FILE, RAY_TUNE_CHECKPOINT
from .matcher import create_siamese_model
from .model import create_fusion_model

logger = logging.getLogger(__name__)


def get_ray_tune_ckpt_callback():
    """
    This is a workaround for the issue caused by the mixed use of old and new lightning's import style.
    https://github.com/optuna/optuna/issues/4689
    We can remove this function after ray adopts the new lightning import style.
    """
    from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

    class _TuneReportCheckpointCallback(TuneReportCheckpointCallback, pl.Callback):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    return _TuneReportCheckpointCallback


def hpo_trial(sampled_hyperparameters, learner, checkpoint_dir=None, **_fit_args):
    """
    Run one HPO trial.

    Parameters
    ----------
    sampled_hyperparameters
        The sampled hyperparameters for this trial.
    learner
        A learner object.
    checkpoint_dir
        The checkpoint directory.
    _fit_args
        The keyword arguments for learner.fit_per_run().
    """
    from ray import train

    context = train.get_context()
    resources = context.get_trial_resources().required_resources
    num_cpus = int(resources.get("CPU"))

    # The original hyperparameters is the search space, replace it with the hyperparameters sampled
    _fit_args["hyperparameters"] = sampled_hyperparameters

    _fit_args["save_path"] = context.get_trial_dir()  # We want to save each trial to a separate directory
    logger.debug(f"hpo trial save_path: {_fit_args['save_path']}")
    if checkpoint_dir is not None:
        _fit_args["resume"] = True
        _fit_args["ckpt_path"] = os.path.join(checkpoint_dir, RAY_TUNE_CHECKPOINT)
    with set_torch_num_threads(num_cpus=num_cpus):
        learner.fit_per_run(**_fit_args)


def build_final_learner(
    learner,
    best_trial_path,
    save_path,
    last_ckpt_path,
    is_matching,
    standalone,
    clean_ckpts,
):
    """
    Build the final learner after HPO is finished.

    Parameters
    ----------
    learner
        A learner object.
    best_trial_path
        The best trial's saving path.
    save_path
        The saving path.
    last_ckpt_path
        The last checkpoint's path.
    is_matching
        Whether is matching.

    Returns
    -------
    The constructed learner.
    """
    if is_matching:
        from ..learners.matching import MultiModalMatcher

        # reload the learner metadata
        matcher = MultiModalMatcher._load_metadata(matcher=learner, path=best_trial_path)
        # construct the model
        matcher._query_model, matcher._response_model = create_siamese_model(
            query_config=matcher._query_config,
            response_config=matcher._response_config,
            pretrained=False,
        )
        # average checkpoint
        matcher.top_k_average(
            save_path=best_trial_path,
            last_ckpt_path=last_ckpt_path,
            top_k_average_method=matcher._config.optimization.top_k_average_method,
        )
        matcher._save_path = save_path

        return matcher
    else:
        from ..learners import BaseLearner

        # reload the learner metadata
        learner = BaseLearner._load_metadata(learner=learner, path=best_trial_path)
        # construct the model
        model = create_fusion_model(
            config=learner._config,
            num_classes=learner._output_shape,
            classes=learner._classes if hasattr(learner, "_classes") else None,
            num_numerical_columns=len(learner._df_preprocessor.numerical_feature_names),
            num_categories=learner._df_preprocessor.categorical_num_categories,
            pretrained=False,  # set "pretrain=False" to prevent downloading online models
        )
        learner._model = model
        # average checkpoint
        learner.top_k_average(
            save_path=best_trial_path,
            last_ckpt_path=last_ckpt_path,
            top_k_average_method=learner._config.optimization.top_k_average_method,
            standalone=standalone,
            clean_ckpts=clean_ckpts,
        )

        learner._save_path = save_path

        return learner


def hyperparameter_tune(hyperparameter_tune_kwargs, resources, is_matching=False, **_fit_args):
    """
    Tune hyperparameters of learner.

    Parameters
    ----------
    hyperparameter_tune_kwargs
        The hyperparameters for HPO, such as, searcher, scheduler, and num_trials.
    resources
        The resources for HPO.
    is_matching
        Whether is matching.
    _fit_args
        The keyword arguments for learner.fit_per_run().

    Returns
    -------
    The learner after tuning hyperparameters.
    """
    from ray.air.config import CheckpointConfig

    from autogluon.core.hpo.ray_hpo import (
        AutommRayTuneAdapter,
        EmptySearchSpace,
        cleanup_checkpoints,
        cleanup_trials,
        run,
    )

    ray_tune_adapter = AutommRayTuneAdapter()
    search_space = _fit_args.get("hyperparameters", dict())
    metric = "val_" + _fit_args.get("learner")._validation_metric_name
    mode = _fit_args.get("learner")._minmax_mode
    save_path = _fit_args.get("save_path")
    time_budget_s = _fit_args.get("max_time")
    num_to_keep = hyperparameter_tune_kwargs.pop("num_to_keep", 3)
    if time_budget_s is not None:
        time_budget_s *= 0.95  # give some buffer time to ray
    try:
        run_config_kwargs = {
            "checkpoint_config": CheckpointConfig(
                num_to_keep=num_to_keep,
                checkpoint_score_attribute=metric,
            ),
        }
        analysis = run(
            trainable=hpo_trial,
            trainable_args=_fit_args,
            search_space=search_space,
            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
            metric=metric,
            mode=mode,
            save_dir=save_path,
            ray_tune_adapter=ray_tune_adapter,
            total_resources=resources,
            minimum_gpu_per_trial=1.0 if resources["num_gpus"] > 0 else 0.0,
            time_budget_s=time_budget_s,
            tune_config_kwargs={"reuse_actors": False},  # reuse_actors cause crashing in ray tune
            run_config_kwargs=run_config_kwargs,
            verbose=2,
        )
    except EmptySearchSpace:
        raise ValueError("Please provide a search space using `hyperparameters` in order to do hyperparameter tune")
    except Exception as e:
        raise e
    else:
        # find the best trial
        best_trial = analysis.get_best_trial(
            metric=metric,
            mode=mode,
        )
        if best_trial is None:
            raise ValueError(
                "MultiModalPredictor wasn't able to find the best trial."
                "Either all trials failed or"
                "it's likely that the time is not enough to train a single epoch for trials."
            )
        # clean up other trials
        logger.info("Removing non-optimal trials and only keep the best one.")
        cleanup_trials(save_path, best_trial.trial_id)
        best_trial_path = os.path.join(save_path, best_trial.trial_id)

        checkpoints_paths_and_scores = dict(
            (os.path.join(checkpoint.path, RAY_TUNE_CHECKPOINT), score)
            for checkpoint, score in analysis._get_trial_checkpoints_with_metric(best_trial, metric=metric)
        )
        # write checkpoint paths and scores to yaml file so that top_k_average could read it
        best_k_model_path = os.path.join(best_trial_path, BEST_K_MODELS_FILE)
        with open(best_k_model_path, "w") as yaml_file:
            yaml.dump(checkpoints_paths_and_scores, yaml_file, default_flow_style=False)

        with analysis.get_last_checkpoint(best_trial).as_directory() as last_ckpt_path:
            learner = build_final_learner(
                learner=_fit_args.get("learner"),
                best_trial_path=best_trial_path,
                save_path=save_path,
                last_ckpt_path=last_ckpt_path,
                is_matching=is_matching,
                standalone=_fit_args.get("standalone"),
                clean_ckpts=_fit_args.get("clean_ckpts"),
            )

        cleanup_checkpoints(best_trial_path)
        # move trial learner one level up
        contents = os.listdir(best_trial_path)
        for content in contents:
            shutil.move(
                os.path.join(best_trial_path, content),
                os.path.join(save_path, content),
            )
        shutil.rmtree(best_trial_path)

        return learner
