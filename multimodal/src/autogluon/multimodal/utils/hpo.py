import logging
import os
import shutil

import yaml

from ..constants import AUTOMM, BEST_K_MODELS_FILE, RAY_TUNE_CHECKPOINT
from .matcher import create_siamese_model
from .model import create_fusion_model

logger = logging.getLogger(AUTOMM)


def hpo_trial(sampled_hyperparameters, predictor, checkpoint_dir=None, **_fit_args):
    """
    Run one HPO trial.

    Parameters
    ----------
    sampled_hyperparameters
        The sampled hyperparameters for this trial.
    predictor
        A predictor object.
    checkpoint_dir
        The checkpoint directory.
    _fit_args
        The keyword arguments for predictor._fit().
    """
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


def build_final_predictor(
    predictor, best_trial_path, minmax_mode, is_distill, val_df, save_path, last_ckpt_path, is_matching
):
    """
    Build the final predictor after HPO is finished.

    Parameters
    ----------
    predictor
        A predictor object.
    best_trial_path
        The best trial's saving path.
    minmax_mode
        min or max.
    is_distill
        Whether is distillation.
    val_df
        Validation dataframe.
    save_path
        The saving path.
    last_ckpt_path
        The last checkpoint's path.
    is_matching
        Whether is matching.

    Returns
    -------
    The constructed predictor.
    """
    if is_matching:
        from ..matcher import MultiModalMatcher

        # reload the predictor metadata
        matcher = MultiModalMatcher._load_metadata(matcher=predictor, path=best_trial_path)
        # construct the model
        query_model, response_model = create_siamese_model(
            query_config=matcher._query_config,
            response_config=matcher._response_config,
            pretrained=False,
        )
        matcher._query_model = query_model
        matcher._response_model = response_model
        # average checkpoint
        matcher._top_k_average(
            query_model=query_model,
            response_model=response_model,
            save_path=best_trial_path,
            last_ckpt_path=last_ckpt_path,
            minmax_mode=minmax_mode,
            top_k_average_method=matcher._config.optimization.top_k_average_method,
            val_df=val_df,
            validation_metric_name=matcher._validation_metric_name,
        )
        matcher._save_path = save_path

        return matcher
    else:
        from ..predictor import MultiModalPredictor

        # reload the predictor metadata
        predictor = MultiModalPredictor._load_metadata(predictor=predictor, path=best_trial_path)
        # construct the model
        model = create_fusion_model(
            config=predictor._config,
            num_classes=predictor._output_shape,
            classes=predictor._classes,
            num_numerical_columns=len(predictor._df_preprocessor.numerical_feature_names),
            num_categories=predictor._df_preprocessor.categorical_num_categories,
            pretrained=False,  # set "pretrain=False" to prevent downloading online models
        )
        predictor._model = model
        # average checkpoint
        predictor._top_k_average(
            model=predictor._model,
            save_path=best_trial_path,
            last_ckpt_path=last_ckpt_path,
            minmax_mode=minmax_mode,
            is_distill=is_distill,
            top_k_average_method=predictor._config.optimization.top_k_average_method,
            val_df=val_df,
            validation_metric_name=predictor._validation_metric_name,
        )

        predictor._save_path = save_path

        return predictor


def hyperparameter_tune(hyperparameter_tune_kwargs, resources, is_matching=False, **_fit_args):
    """
    Tune hyperparameters of predictor.

    Parameters
    ----------
    hyperparameter_tune_kwargs
        The hyperparameters for HPO, such as, searcher, scheduler, and num_trials.
    resources
        The resources for HPO.
    is_matching
        Whether is matching.
    _fit_args
        The keyword arguments for predictor._fit().

    Returns
    -------
    The predictor after tuning hyperparameters.
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
    metric = "val_" + _fit_args.get("validation_metric_name")
    mode = _fit_args.get("minmax_mode")
    save_path = _fit_args.get("save_path")
    time_budget_s = _fit_args.get("max_time")
    is_distill = False
    if _fit_args.get("teacher_predictor", None) is not None:
        is_distill = True
    try:
        run_config_kwargs = {
            "checkpoint_config": CheckpointConfig(
                num_to_keep=3,
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
            (os.path.join(checkpoint, RAY_TUNE_CHECKPOINT), score)
            for checkpoint, score in analysis.get_trial_checkpoints_paths(best_trial, metric=metric)
        )
        # write checkpoint paths and scores to yaml file so that top_k_average could read it
        best_k_model_path = os.path.join(best_trial_path, BEST_K_MODELS_FILE)
        with open(best_k_model_path, "w") as yaml_file:
            yaml.dump(checkpoints_paths_and_scores, yaml_file, default_flow_style=False)

        with analysis.get_last_checkpoint(best_trial).as_directory() as last_ckpt_path:
            predictor = build_final_predictor(
                predictor=_fit_args.get("predictor"),
                best_trial_path=best_trial_path,
                minmax_mode=mode,
                is_distill=is_distill,
                val_df=_fit_args["val_df"],
                save_path=save_path,
                last_ckpt_path=last_ckpt_path,
                is_matching=is_matching,
            )

        cleanup_checkpoints(best_trial_path)
        # move trial predictor one level up
        contents = os.listdir(best_trial_path)
        for content in contents:
            shutil.move(
                os.path.join(best_trial_path, content),
                os.path.join(save_path, content),
            )
        shutil.rmtree(best_trial_path)

        return predictor
