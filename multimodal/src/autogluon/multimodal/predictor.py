from __future__ import annotations

import copy
import json
import logging
import math
import operator
import os
import pickle
import shutil
import sys
import warnings
from datetime import timedelta
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from packaging import version
from sklearn.model_selection import train_test_split
from torch import nn

from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.common.utils.utils import setup_outputdir
from autogluon.core.utils.try_import import try_import_ray_lightning
from autogluon.core.utils.utils import default_holdout_frac

from . import version as ag_version
from .constants import (
    AUTOMM,
    AUTOMM_TUTORIAL_MODE,
    BBOX,
    BEST,
    BEST_K_MODELS_FILE,
    BINARY,
    CLASSIFICATION,
    COLUMN_FEATURES,
    DATA,
    DEEPSPEED_MIN_PL_VERSION,
    DEEPSPEED_MODULE,
    DEEPSPEED_OFFLOADING,
    DEEPSPEED_STRATEGY,
    DEPRECATED_ZERO_SHOT,
    FEATURE_EXTRACTION,
    FEATURES,
    FEW_SHOT,
    GREEDY_SOUP,
    LABEL,
    LAST_CHECKPOINT,
    LOGITS,
    MAP,
    MASKS,
    MAX,
    MIN,
    MODEL,
    MODEL_CHECKPOINT,
    MULTICLASS,
    NER,
    NER_RET,
    OBJECT_DETECTION,
    OCR_TEXT_DETECTION,
    OCR_TEXT_RECOGNITION,
    PROBABILITY,
    RAY_TUNE_CHECKPOINT,
    REGRESSION,
    SCORE,
    TEXT,
    UNIFORM_SOUP,
    Y_PRED,
    Y_PRED_PROB,
    Y_TRUE,
    ZERO_SHOT_IMAGE_CLASSIFICATION,
)
from .data.datamodule import BaseDataModule
from .data.infer_types import (
    infer_column_types,
    infer_label_column_type_by_problem_type_and_pipeline,
    infer_problem_type_output_shape,
    infer_rois_column_type,
)
from .data.preprocess_dataframe import MultiModalFeaturePreprocessor
from .data.utils import apply_data_processor, apply_df_preprocessor, get_collate_fn, get_per_sample_features
from .models.utils import get_model_postprocess_fn
from .optimization.lit_distiller import DistillerLitModule
from .optimization.lit_mmdet import MMDetLitModule
from .optimization.lit_module import LitModule
from .optimization.lit_ner import NerLitModule
from .optimization.losses import RKDLoss
from .optimization.utils import (
    get_loss_func,
    get_metric,
    get_norm_layer_param_names,
    get_trainable_params_efficient_finetune,
)
from .utils import (
    AutoMMModelCheckpoint,
    AutoMMModelCheckpointIO,
    COCODataset,
    CustomUnpickler,
    LogFilter,
    apply_log_filter,
    assign_feature_column_names,
    average_checkpoints,
    cocoeval,
    compute_inference_batch_size,
    compute_num_gpus,
    compute_score,
    create_fusion_data_processors,
    create_fusion_model,
    data_to_df,
    extract_from_output,
    filter_search_space,
    from_coco,
    get_config,
    get_local_pretrained_config_paths,
    get_minmax_mode,
    get_mixup,
    get_onnx_input,
    getCOCOCatIDs,
    hpo_trial,
    infer_batch,
    infer_dtypes_by_model_names,
    infer_metrics,
    infer_precision,
    infer_scarcity_mode_by_data_size,
    init_df_preprocessor,
    init_pretrained,
    load_text_tokenizers,
    logits_to_prob,
    modify_duplicate_model_names,
    process_save_path,
    save_pretrained_model_configs,
    save_text_tokenizers,
    select_model,
    tensor_to_ndarray,
    try_to_infer_pos_label,
    turn_on_off_feature_column_info,
    update_config_by_rules,
    use_realtime,
)

logger = logging.getLogger(AUTOMM)


class MultiModalPredictor:
    """
    MultiModalPredictor is a deep learning "model zoo" of model zoos. It can automatically build deep learning models that
    are suitable for multimodal datasets. You will only need to preprocess the data in the multimodal dataframe format
    and the MultiModalPredictor can predict the values of one column conditioned on the features from the other columns.

    The prediction can be either classification or regression. The feature columns can contain
    image paths, text, numerical, and categorical values.

    """

    def __init__(
        self,
        label: Optional[str] = None,
        problem_type: Optional[str] = None,
        pipeline: Optional[str] = None,
        val_metric: Optional[str] = None,
        eval_metric: Optional[str] = None,
        hyperparameters: Optional[dict] = None,
        path: Optional[str] = None,
        verbosity: Optional[int] = 3,
        output_shape: Optional[int] = None,  # TODO: infer this for detection
        warn_if_exist: Optional[bool] = True,
        enable_progress_bar: Optional[bool] = None,
    ):
        """
        Parameters
        ----------
        label
            Name of the column that contains the target variable to predict.
        problem_type
            Type of prediction problem, i.e. is this a binary/multiclass classification or regression problem
            (options: 'binary', 'multiclass', 'regression').
            If `problem_type = None`, the prediction problem type is inferred
            based on the label-values in provided dataset.
        pipeline
            This defines inference tasks like FeatureExtraction, ZeroShotClassification, etc.
            TODO: add more pipelines (ref: https://huggingface.co/docs/transformers/main_classes/pipelines)
        eval_metric
            Evaluation metric name. If `eval_metric = None`, it is automatically chosen based on `problem_type`.
            Defaults to 'accuracy' for binary and multiclass classification, 'root_mean_squared_error' for regression.
        hyperparameters
            This is to override some default configurations.
            For example, changing the text and image backbones can be done by formatting:

            a string
            hyperparameters = "model.hf_text.checkpoint_name=google/electra-small-discriminator model.timm_image.checkpoint_name=swin_small_patch4_window7_224"

            or a list of strings
            hyperparameters = ["model.hf_text.checkpoint_name=google/electra-small-discriminator", "model.timm_image.checkpoint_name=swin_small_patch4_window7_224"]

            or a dictionary
            hyperparameters = {
                            "model.hf_text.checkpoint_name": "google/electra-small-discriminator",
                            "model.timm_image.checkpoint_name": "swin_small_patch4_window7_224",
                        }
        path
            Path to directory where models and intermediate outputs should be saved.
            If unspecified, a time-stamped folder called "AutogluonAutoMM/ag-[TIMESTAMP]"
            will be created in the working directory to store all models.
            Note: To call `fit()` twice and save all results of each fit,
            you must specify different `path` locations or don't specify `path` at all.
            Otherwise files from first `fit()` will be overwritten by second `fit()`.
        verbosity
            Verbosity levels range from 0 to 4 and control how much information is printed.
            Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
            If using logging, you can alternatively control amount of information printed via `logger.setLevel(L)`,
            where `L` ranges from 0 to 50
            (Note: higher values of `L` correspond to fewer print statements, opposite of verbosity levels)
        warn_if_exist
            Whether to raise warning if the specified path already exists.
        enable_progress_bar
            Whether to show progress bar. It will be True by default and will also be
            disabled if the environment variable os.environ["AUTOMM_DISABLE_PROGRESS_BAR"] is set.
        """
        if eval_metric is not None and not isinstance(eval_metric, str):
            eval_metric = eval_metric.name

        if eval_metric is not None and eval_metric.lower() in [
            "rmse",
            "r2",
            "pearsonr",
            "spearmanr",
        ]:
            problem_type = REGRESSION

        if os.environ.get(AUTOMM_TUTORIAL_MODE):
            verbosity = 1  # don't use 3, which doesn't suppress logger.info() in .load().
            enable_progress_bar = False

        if verbosity is not None:
            set_logger_verbosity(verbosity, logger=logger)

        if path is not None:
            path = process_save_path(path=path)

        self._label_column = label
        self._problem_type = problem_type.lower() if problem_type is not None else None
        self._pipeline = pipeline.lower() if pipeline is not None else None
        self._eval_metric_name = eval_metric
        self._validation_metric_name = val_metric
        self._output_shape = output_shape
        self._save_path = path
        self._ckpt_path = None
        self._pretrained_path = None
        self._config = None
        self._df_preprocessor = None
        self._column_types = None
        self._data_processors = None
        self._model_postprocess_fn = None
        self._model = None
        self._resume = False
        self._continuous_training = False
        self._verbosity = verbosity
        self._warn_if_exist = warn_if_exist
        self._enable_progress_bar = enable_progress_bar if enable_progress_bar is not None else True

        if problem_type is not None and problem_type.lower() == DEPRECATED_ZERO_SHOT:
            warnings.warn(
                f'problem_type="{problem_type}" is deprecated. For inference with CLIP model, '
                f'use pipeline="zero_shot_image_classification" instead.',
                DeprecationWarning,
            )
            self._problem_type = None
            self._pipeline = ZERO_SHOT_IMAGE_CLASSIFICATION

        if problem_type is not None and problem_type.lower() == NER:
            self._pipeline = None

        if self._pipeline is not None:
            self._config, self._model, self._data_processors = init_pretrained(
                pipeline=self._pipeline, hyperparameters=hyperparameters, num_classes=self._output_shape
            )

    @property
    def path(self):
        return self._save_path

    @property
    def label(self):
        return self._label_column

    @property
    def problem_type(self):
        return self._problem_type

    @property
    def column_types(self):
        return self._column_types

    # This func is required by the abstract trainer of TabularPredictor.
    def set_verbosity(self, verbosity: int):
        """Set the verbosity level of the log.

        Parameters
        ----------
        verbosity
            The verbosity level

        """
        self._verbosity = verbosity
        set_logger_verbosity(verbosity, logger=logger)

    def fit(
        self,
        train_data: Union[pd.DataFrame, str],
        presets: Optional[str] = None,
        config: Optional[dict] = None,
        tuning_data: Optional[Union[pd.DataFrame, str]] = None,
        time_limit: Optional[int] = None,
        save_path: Optional[str] = None,
        hyperparameters: Optional[Union[str, Dict, List[str]]] = None,
        column_types: Optional[dict] = None,
        holdout_frac: Optional[float] = None,
        teacher_predictor: Union[str, MultiModalPredictor] = None,
        seed: Optional[int] = 123,
        standalone: Optional[bool] = True,
        hyperparameter_tune_kwargs: Optional[dict] = None,
    ):
        """
        Fit MultiModalPredictor predict label column of a dataframe based on the other columns,
        which may contain image path, text, numeric, or categorical features.

        Parameters
        ----------
        train_data
            A dataframe containing training data.
        presets
            Name of the presets. See the available presets in `presets.py`.
        config
            A dictionary with four keys "model", "data", "optimization", and "environment".
            Each key's value can be a string, yaml file path, or OmegaConf's DictConfig.
            Strings should be the file names (DO NOT include the postfix ".yaml") in
            automm/configs/model, automm/configs/data, automm/configs/optimization, and automm/configs/environment.
            For example, you can configure a late-fusion model for the image, text, and tabular data as follows:
            config = {
                        "model": "fusion_mlp_image_text_tabular",
                        "data": "default",
                        "optimization": "adamw",
                        "environment": "default",
                    }
            or
            config = {
                        "model": "/path/to/model/config.yaml",
                        "data": "/path/to/data/config.yaml",
                        "optimization": "/path/to/optimization/config.yaml",
                        "environment": "/path/to/environment/config.yaml",
                    }
            or
            config = {
                        "model": OmegaConf.load("/path/to/model/config.yaml"),
                        "data": OmegaConf.load("/path/to/data/config.yaml"),
                        "optimization": OmegaConf.load("/path/to/optimization/config.yaml"),
                        "environment": OmegaConf.load("/path/to/environment/config.yaml"),
                    }
        tuning_data
            A dataframe containing validation data, which should have the same columns as the train_data.
            If `tuning_data = None`, `fit()` will automatically
            hold out some random validation examples from `train_data`.
        time_limit
            How long `fit()` should run for (wall clock time in seconds).
            If not specified, `fit()` will run until the model has completed training.
        save_path
            Path to directory where models and intermediate outputs should be saved.
        hyperparameters
            This is to override some default configurations.
            For example, changing the text and image backbones can be done by formatting:

            a string
            hyperparameters = "model.hf_text.checkpoint_name=google/electra-small-discriminator model.timm_image.checkpoint_name=swin_small_patch4_window7_224"

            or a list of strings
            hyperparameters = ["model.hf_text.checkpoint_name=google/electra-small-discriminator", "model.timm_image.checkpoint_name=swin_small_patch4_window7_224"]

            or a dictionary
            hyperparameters = {
                            "model.hf_text.checkpoint_name": "google/electra-small-discriminator",
                            "model.timm_image.checkpoint_name": "swin_small_patch4_window7_224",
                        }
        column_types
            A dictionary that maps column names to their data types.
            For example: `column_types = {"item_name": "text", "image": "image_path",
            "product_description": "text", "height": "numerical"}`
            may be used for a table with columns: "item_name", "brand", "product_description", and "height".
            If None, column_types will be automatically inferred from the data.
            The current supported types are:
                - "image_path": each row in this column is one image path.
                - "text": each row in this column contains text (sentence, paragraph, etc.).
                - "numerical": each row in this column contains a number.
                - "categorical": each row in this column belongs to one of K categories.
        holdout_frac
            Fraction of train_data to holdout as tuning_data for optimizing hyper-parameters or
            early stopping (ignored unless `tuning_data = None`).
            Default value (if None) is selected based on the number of rows in the training data
            and whether hyper-parameter-tuning is utilized.
        teacher_predictor
            The pre-trained teacher predictor or its saved path. If provided, `fit()` can distill its
            knowledge to a student predictor, i.e., the current predictor.
        seed
            The random seed to use for this training run.
        standalone
            Whether to save the enire model for offline deployment or only trained parameters of parameter-efficient fine-tuning strategy.
        hyperparameter_tune_kwargs
                Hyperparameter tuning strategy and kwargs (for example, how many HPO trials to run).
                If None, then hyperparameter tuning will not be performed.
                    num_trials: int
                        How many HPO trials to run. Either `num_trials` or `time_limit` to `fit` needs to be specified.
                    scheduler: Union[str, ray.tune.schedulers.TrialScheduler]
                        If str is passed, AutoGluon will create the scheduler for you with some default parameters.
                        If ray.tune.schedulers.TrialScheduler object is passed, you are responsible for initializing the object.
                    scheduler_init_args: Optional[dict] = None
                        If provided str to `scheduler`, you can optionally provide custom init_args to the scheduler
                    searcher: Union[str, ray.tune.search.SearchAlgorithm, ray.tune.search.Searcher]
                        If str is passed, AutoGluon will create the searcher for you with some default parameters.
                        If ray.tune.schedulers.TrialScheduler object is passed, you are responsible for initializing the object.
                        You don't need to worry about `metric` and `mode` of the searcher object. AutoGluon will figure it out by itself.
                    scheduler_init_args: Optional[dict] = None
                        If provided str to `searcher`, you can optionally provide custom init_args to the searcher
                        You don't need to worry about `metric` and `mode`. AutoGluon will figure it out by itself.

        Returns
        -------
        An "MultiModalPredictor" object (itself).
        """
        if self._pipeline == OBJECT_DETECTION:
            self.detection_anno_train = train_data
            train_data = from_coco(train_data)
            if tuning_data is not None:
                tuning_data = from_coco(tuning_data)

        if hyperparameter_tune_kwargs is not None:
            # TODO: can we support hyperparameters being the same format as regular training?
            # currently the string format would make it very hard to get search space, which is an object
            assert isinstance(
                hyperparameters, dict
            ), "Please provide hyperparameters as a dictionary if you want to do HPO"
            if teacher_predictor is not None:
                assert isinstance(
                    teacher_predictor, str
                ), "HPO with distillation only supports passing a path to the predictor"
            if self._continuous_training:
                warnings.warn(
                    "HPO while continuous training."
                    "Hyperparameters related to Model and Data will NOT take effect."
                    "We will filter them out from the search space."
                )
                hyperparameters = filter_search_space(hyperparameters, [MODEL, DATA])

        pl.seed_everything(seed, workers=True)

        if self._resume:
            assert hyperparameter_tune_kwargs is None, "You can not resume training with HPO"
            save_path = process_save_path(path=self._save_path, resume=True)
        elif save_path is not None:
            save_path = process_save_path(path=save_path)
        elif self._save_path is not None:
            save_path = process_save_path(path=self._save_path, raise_if_exist=False)

        if not self._resume:
            save_path = setup_outputdir(
                path=save_path,
                warn_if_exist=self._warn_if_exist,
            )

        save_path = os.path.abspath(os.path.expanduser(save_path))
        self._save_path = save_path
        logger.debug(f"save path: {save_path}")

        # Generate general info that's not config specific
        if tuning_data is None:
            if self._problem_type in [BINARY, MULTICLASS, CLASSIFICATION]:
                stratify = train_data[self._label_column]
            else:
                stratify = None
            if holdout_frac is None:
                val_frac = default_holdout_frac(len(train_data), hyperparameter_tune=False)
            else:
                val_frac = holdout_frac
            train_data, tuning_data = train_test_split(
                train_data,
                test_size=val_frac,
                stratify=stratify,
                random_state=np.random.RandomState(seed),
            )

        column_types = infer_column_types(
            data=train_data,
            valid_data=tuning_data,
            label_columns=self._label_column,
            provided_column_types=column_types,
        )
        column_types = infer_label_column_type_by_problem_type_and_pipeline(
            column_types=column_types,
            label_columns=self._label_column,
            problem_type=self._problem_type,
            pipeline=self._pipeline,
            data=train_data,
            valid_data=tuning_data,
        )

        if self._config is not None:  # continuous training
            config = self._config

        problem_type, output_shape = infer_problem_type_output_shape(
            label_column=self._label_column,
            column_types=column_types,
            data=train_data,
            provided_problem_type=self._problem_type,
            pipeline=self._pipeline,
        )

        # Determine data scarcity mode, i.e. a few-shot scenario
        scarcity_mode = infer_scarcity_mode_by_data_size(
            df_train=train_data, scarcity_threshold=50
        )  # Add as separate hyperparameter somewhere?
        if scarcity_mode == FEW_SHOT and (not presets or FEW_SHOT not in presets):  # TODO: check for data  type
            logger.info(
                f"Detected data scarcity. Consider running using the preset 'few_shot_text_classification' for better performance."
            )

        logger.debug(f"column_types: {column_types}")
        logger.debug(f"image columns: {[k for k, v in column_types.items() if v == 'image_path']}")

        if self._column_types is not None and self._column_types != column_types:
            warnings.warn(
                f"Inferred column types {column_types} are inconsistent with "
                f"the previous {self._column_types}. "
                f"New columns will not be used in the current training."
            )
            # use previous column types to avoid inconsistency with previous numerical mlp and categorical mlp
            column_types = self._column_types

        if self._problem_type is not None:
            if self._problem_type == CLASSIFICATION:
                # Set the problem type to be inferred problem type
                self._problem_type = problem_type
            assert self._problem_type == problem_type, (
                f"Inferred problem type {problem_type} is different from " f"the previous {self._problem_type}"
            )

        if self._pipeline != OBJECT_DETECTION:
            if self._output_shape is not None:
                assert self._output_shape == output_shape, (
                    f"Inferred output shape {output_shape} is different from " f"the previous {self._output_shape}"
                )
            else:
                self._output_shape = output_shape

        if self._validation_metric_name is None or self._eval_metric_name is None:
            validation_metric_name, eval_metric_name = infer_metrics(
                problem_type=problem_type,
                pipeline=self._pipeline,
                eval_metric_name=self._eval_metric_name,
                validation_metric_name=self._validation_metric_name,
            )
        else:
            validation_metric_name = self._validation_metric_name
            eval_metric_name = self._eval_metric_name
        minmax_mode = get_minmax_mode(validation_metric_name)

        if time_limit is not None:
            time_limit = timedelta(seconds=time_limit)

        # set attributes for saving and prediction
        self._problem_type = problem_type  # In case problem type isn't provided in __init__().
        self._eval_metric_name = eval_metric_name  # In case eval_metric isn't provided in __init__().
        self._validation_metric_name = validation_metric_name
        self._save_path = save_path
        self._column_types = column_types

        _fit_args = dict(
            train_df=train_data,
            val_df=tuning_data,
            validation_metric_name=validation_metric_name,
            minmax_mode=minmax_mode,
            max_time=time_limit,
            save_path=save_path,
            ckpt_path=None if hyperparameter_tune_kwargs is not None else self._ckpt_path,
            resume=False if hyperparameter_tune_kwargs is not None else self._resume,
            enable_progress_bar=False if hyperparameter_tune_kwargs is not None else self._enable_progress_bar,
            presets=presets,
            config=config,
            hyperparameters=hyperparameters,
            teacher_predictor=teacher_predictor,
            standalone=standalone,
            hpo_mode=(hyperparameter_tune_kwargs is not None),  # skip average checkpoint if in hpo mode
        )

        if hyperparameter_tune_kwargs is not None:
            # TODO: allow custom gpu
            resources = dict(num_gpus=torch.cuda.device_count())
            if _fit_args["max_time"] is not None:
                _fit_args["max_time"] *= 0.95  # give some buffer time to ray lightning trainer
            _fit_args["predictor"] = self
            predictor = self._hyperparameter_tune(
                hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                resources=resources,
                **_fit_args,
            )
            return predictor

        self._fit(**_fit_args)
        return self

    def _hyperparameter_tune(self, hyperparameter_tune_kwargs, resources, **_fit_args):
        from ray.air.config import CheckpointConfig

        from autogluon.core.hpo.ray_hpo import (
            AutommRayTuneAdapter,
            AutommRayTuneLightningAdapter,
            EmptySearchSpace,
            cleanup_checkpoints,
            cleanup_trials,
            run,
        )

        ray_tune_adapter = AutommRayTuneAdapter()
        if try_import_ray_lightning():
            ray_tune_adapter = AutommRayTuneLightningAdapter()
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
            raise ValueError(
                "Please provide a search space using `hyperparameters` in order to do hyperparameter tune"
            )
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
            # reload the predictor metadata
            predictor = MultiModalPredictor._load_metadata(predictor=self, path=best_trial_path)
            # construct the model
            model = create_fusion_model(
                config=predictor._config,
                num_classes=predictor._output_shape,
                num_numerical_columns=len(predictor._df_preprocessor.numerical_feature_names),
                num_categories=predictor._df_preprocessor.categorical_num_categories,
                pretrained=False,  # set "pretrain=False" to prevent downloading online models
            )
            predictor._model = model
            # average checkpoint
            checkpoints_paths_and_scores = dict(
                (os.path.join(checkpoint, RAY_TUNE_CHECKPOINT), score)
                for checkpoint, score in analysis.get_trial_checkpoints_paths(best_trial, metric=metric)
            )
            # write checkpoint paths and scores to yaml file so that top_k_average could read it
            best_k_model_path = os.path.join(best_trial_path, BEST_K_MODELS_FILE)
            with open(best_k_model_path, "w") as yaml_file:
                yaml.dump(checkpoints_paths_and_scores, yaml_file, default_flow_style=False)

            with analysis.get_last_checkpoint(best_trial).as_directory() as last_ckpt_dir:
                predictor._top_k_average(
                    model=predictor._model,
                    save_path=best_trial_path,
                    last_ckpt_path=last_ckpt_dir,
                    minmax_mode=mode,
                    is_distill=is_distill,
                    top_k_average_method=predictor._config.optimization.top_k_average_method,
                    val_df=_fit_args["val_df"],
                    validation_metric_name=predictor._validation_metric_name,
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
            predictor._save_path = save_path

            return predictor

    def _setup_distillation(
        self,
        teacher_predictor: Union[str, MultiModalPredictor],
    ):
        """
        Prepare for distillation. It verifies whether the student and teacher predictors have consistent
        configurations. If teacher and student have duplicate model names, it modifies teacher's model names.

        Parameters
        ----------
        teacher_predictor
            The teacher predictor in knowledge distillation.

        Returns
        -------
        teacher_model
            The teacher predictor's model.
        critics
            The critics used in computing mutual information loss.
        baseline_funcs
            The baseline functions used in computing mutual information loss.
        soft_label_loss_func
            The loss function using teacher's logits as labels.
        output_feature_adaptor
            The adaptor used to adapt student output feature to the shape of teacher's.
        output_feature_loss_func
            The loss function using minimize distance between output_feature of teacher and student.
        rkd_loss_func
            The loss function using rkd distance and angle loss between output_feature of teacher and student.
        df_preprocessor
            The teacher predictor's dataframe preprocessor.
        data_processors
            The teacher predictor's data processors.
        """
        logger.debug("setting up distillation...")
        if isinstance(teacher_predictor, str):
            teacher_predictor = MultiModalPredictor.load(teacher_predictor)

        # verify that student and teacher configs are consistent.
        assert self._problem_type == teacher_predictor._problem_type
        assert self._label_column == teacher_predictor._label_column
        assert self._eval_metric_name == teacher_predictor._eval_metric_name
        assert self._output_shape == teacher_predictor._output_shape
        assert self._validation_metric_name == teacher_predictor._validation_metric_name

        # if teacher and student have duplicate model names, change teacher's model names
        # we don't change student's model names to avoid changing the names back when saving the model.
        teacher_predictor = modify_duplicate_model_names(
            predictor=teacher_predictor,
            postfix="teacher",
            blacklist=self._config.model.names,
        )

        critics, baseline_funcs = None, None
        if not self._config.distiller.soft_label_loss_type:
            # automatically infer loss func based on problem type if not specified
            if self._problem_type == "regression":
                soft_label_loss_func = nn.MSELoss()
            else:
                assert self._output_shape > 1
                soft_label_loss_func = nn.CrossEntropyLoss()
        elif self._config.distiller.soft_label_loss_type == "mse":
            soft_label_loss_func = nn.MSELoss()
        elif self._config.distiller.soft_label_loss_type == "cross_entropy":
            soft_label_loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown soft_label_loss_type: {self._config.distiller.soft_label_loss_type}")

        if not self._config.distiller.softmax_regression_loss_type:
            # automatically infer loss func based on problem type if not specified
            if self._problem_type == "regression":
                softmax_regression_loss_func = nn.MSELoss()
            else:
                assert self._output_shape > 1
                softmax_regression_loss_func = nn.CrossEntropyLoss()
        elif self._config.distiller.softmax_regression_loss_type == "mse":
            softmax_regression_loss_func = nn.MSELoss()
        elif self._config.distiller.softmax_regression_loss_type == "cross_entropy":
            softmax_regression_loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown soft_label_loss_type: {self._config.distiller.softmax_regression_loss_type}")

        output_feature_loss_type = OmegaConf.select(self._config, "distiller.output_feature_loss_type", default="mse")
        if output_feature_loss_type == "cosine":
            output_feature_loss_func = nn.CosineEmbeddingLoss()
        elif output_feature_loss_type == "mse":
            output_feature_loss_func = nn.MSELoss()
        else:
            raise ValueError(f"Unknown output_feature_loss_type: {output_feature_loss_type}")

        # Adapt student's output_feature feature to teacher's
        # Refer to FitNet: https://arxiv.org/abs/1412.6550
        teacher_model_dim = teacher_predictor._model.out_features
        student_model_dim = self._model.out_features
        output_feature_adaptor = (
            nn.Linear(student_model_dim, teacher_model_dim)
            if teacher_model_dim != student_model_dim
            else nn.Identity()
        )

        rkd_distance_loss_weight = OmegaConf.select(self._config, "distiller.rkd_distance_loss_weight", default=0.0)
        rkd_angle_loss_weight = OmegaConf.select(self._config, "distiller.rkd_angle_loss_weight", default=0.0)
        rkd_loss_func = RKDLoss(rkd_distance_loss_weight, rkd_angle_loss_weight)

        # turn on returning column information in data processors
        turn_on_off_feature_column_info(
            data_processors=self._data_processors,
            flag=True,
        )
        turn_on_off_feature_column_info(
            data_processors=teacher_predictor._data_processors,
            flag=True,
        )

        return (
            teacher_predictor._model,
            critics,
            baseline_funcs,
            soft_label_loss_func,
            softmax_regression_loss_func,
            output_feature_adaptor,
            output_feature_loss_func,
            rkd_loss_func,
            teacher_predictor._df_preprocessor,
            teacher_predictor._data_processors,
        )

    def _fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        validation_metric_name: str,
        minmax_mode: str,
        max_time: timedelta,
        save_path: str,
        ckpt_path: str,
        resume: bool,
        enable_progress_bar: bool,
        presets: Optional[str] = None,
        config: Optional[dict] = None,
        hyperparameters: Optional[Union[str, Dict, List[str]]] = None,
        teacher_predictor: Union[str, MultiModalPredictor] = None,
        hpo_mode: bool = False,
        standalone: bool = True,
        **hpo_kwargs,
    ):
        if self._config is not None:  # continuous training
            config = self._config

        config = get_config(
            presets=presets,
            config=config,
            overrides=hyperparameters,
            extra=["distiller"] if teacher_predictor is not None else None,
        )

        config = update_config_by_rules(
            problem_type=self._problem_type,
            config=config,
        )

        if self._df_preprocessor is None:
            df_preprocessor = init_df_preprocessor(
                config=config,
                column_types=self._column_types,
                label_column=self._label_column,
                train_df_x=train_df.drop(columns=self._label_column),
                train_df_y=train_df[self._label_column],
            )
        else:  # continuing training
            df_preprocessor = self._df_preprocessor

        config = select_model(config=config, df_preprocessor=df_preprocessor)

        if self._model is None:
            model = create_fusion_model(
                config=config,
                num_classes=self._output_shape,
                num_numerical_columns=len(df_preprocessor.numerical_feature_names),
                num_categories=df_preprocessor.categorical_num_categories,
            )
        else:  # continuing training
            model = self._model

        norm_param_names = get_norm_layer_param_names(model)

        trainable_param_names = get_trainable_params_efficient_finetune(
            norm_param_names, efficient_finetune=OmegaConf.select(config, "optimization.efficient_finetune")
        )

        if self._data_processors is None:
            data_processors = create_fusion_data_processors(
                config=config,
                model=model,
            )
        else:  # continuing training
            data_processors = self._data_processors

        data_processors_count = {k: len(v) for k, v in data_processors.items()}
        logger.debug(f"data_processors_count: {data_processors_count}")

        pos_label = try_to_infer_pos_label(
            data_config=config.data,
            label_encoder=df_preprocessor.label_generator,
            problem_type=self._problem_type,
        )
        if validation_metric_name is not None:
            validation_metric, custom_metric_func = get_metric(
                metric_name=validation_metric_name,
                num_classes=self._output_shape,
                pos_label=pos_label,
            )
        else:
            validation_metric, custom_metric_func = (None, None)

        mixup_active, mixup_fn = get_mixup(
            model_config=OmegaConf.select(config, "model"),
            mixup_config=OmegaConf.select(config, "data.mixup"),
            num_classes=self._output_shape,
        )
        if mixup_active and (config.env.per_gpu_batch_size == 1 or config.env.per_gpu_batch_size % 2 == 1):
            warnings.warn(
                "The mixup is done on the batch."
                "The per_gpu_batch_size should be >1 and even for reasonable operation",
                UserWarning,
            )

        loss_func = get_loss_func(
            problem_type=self._problem_type,
            mixup_active=mixup_active,
            loss_func_name=OmegaConf.select(config, "optimization.loss_function"),
        )

        model_postprocess_fn = get_model_postprocess_fn(
            problem_type=self._problem_type,
            loss_func=loss_func,
        )

        self._config = config
        self._df_preprocessor = df_preprocessor
        self._data_processors = data_processors
        self._model = model
        self._model_postprocess_fn = model_postprocess_fn

        if max_time == timedelta(seconds=0):
            self._top_k_average(
                model=model,
                save_path=save_path,
                minmax_mode=minmax_mode,
                is_distill=False,
                top_k_average_method=config.optimization.top_k_average_method,
                val_df=val_df,
                validation_metric_name=validation_metric_name,
                strict_loading=not trainable_param_names,
                standalone=standalone,
            )

            return self

        # need to assign the above attributes before setting up distillation
        if teacher_predictor is not None:
            (
                teacher_model,
                critics,
                baseline_funcs,
                soft_label_loss_func,
                softmax_regression_loss_func,
                output_feature_adaptor,
                output_feature_loss_func,
                rkd_loss_func,
                teacher_df_preprocessor,
                teacher_data_processors,
            ) = self._setup_distillation(
                teacher_predictor=teacher_predictor,
            )
        else:
            (
                teacher_model,
                critics,
                baseline_funcs,
                soft_label_loss_func,
                softmax_regression_loss_func,
                output_feature_adaptor,
                output_feature_loss_func,
                rkd_loss_func,
                teacher_df_preprocessor,
                teacher_data_processors,
            ) = (None, None, None, None, None, None, None, None, None, None)

        if teacher_df_preprocessor is not None:
            df_preprocessor = [df_preprocessor, teacher_df_preprocessor]
        if teacher_data_processors is not None:
            data_processors = [data_processors, teacher_data_processors]

        val_use_training_mode = (self._pipeline == OBJECT_DETECTION) and (validation_metric_name != MAP)
        train_dm = BaseDataModule(
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
            per_gpu_batch_size=config.env.per_gpu_batch_size,
            num_workers=config.env.num_workers,
            train_data=train_df,
            val_data=val_df,
            val_use_training_mode=val_use_training_mode,
        )
        optimization_kwargs = dict(
            optim_type=config.optimization.optim_type,
            lr_choice=config.optimization.lr_choice,
            lr_schedule=config.optimization.lr_schedule,
            lr=config.optimization.learning_rate,
            lr_decay=config.optimization.lr_decay,
            end_lr=config.optimization.end_lr,
            lr_mult=config.optimization.lr_mult,
            weight_decay=config.optimization.weight_decay,
            warmup_steps=config.optimization.warmup_steps,
        )
        metrics_kwargs = dict(
            validation_metric=validation_metric,
            validation_metric_name=validation_metric_name,
            custom_metric_func=custom_metric_func,
        )
        is_distill = teacher_model is not None
        if is_distill:
            output_feature_loss_weight = OmegaConf.select(
                self._config, "distiller.output_feature_loss_weight", default=0.0
            )
            softmax_regression_weight = OmegaConf.select(
                self._config, "distiller.softmax_regression_weight", default=0.0
            )
            use_raw_features = OmegaConf.select(self._config, "distiller.use_raw_features", default=False)
            task = DistillerLitModule(
                student_model=model,
                teacher_model=teacher_model,
                matches=config.distiller.matches,
                critics=critics,
                baseline_funcs=baseline_funcs,
                hard_label_weight=config.distiller.hard_label_weight,
                soft_label_weight=config.distiller.soft_label_weight,
                softmax_regression_weight=softmax_regression_weight,
                temperature=config.distiller.temperature,
                output_feature_loss_weight=output_feature_loss_weight,
                hard_label_loss_func=loss_func,
                soft_label_loss_func=soft_label_loss_func,
                softmax_regression_loss_func=softmax_regression_loss_func,
                output_feature_adaptor=output_feature_adaptor,
                output_feature_loss_func=output_feature_loss_func,
                rkd_loss_func=rkd_loss_func,
                **metrics_kwargs,
                **optimization_kwargs,
            )
        elif self._problem_type == NER:
            task = NerLitModule(
                model=model,
                loss_func=loss_func,
                efficient_finetune=OmegaConf.select(config, "optimization.efficient_finetune"),
                mixup_fn=mixup_fn,
                mixup_off_epoch=OmegaConf.select(config, "data.mixup.turn_off_epoch"),
                model_postprocess_fn=model_postprocess_fn,
                trainable_param_names=trainable_param_names,
                **metrics_kwargs,
                **optimization_kwargs,
            )
        elif self._pipeline == OBJECT_DETECTION:
            task = MMDetLitModule(
                model=model,
                **metrics_kwargs,
                **optimization_kwargs,
            )
        else:
            task = LitModule(
                model=model,
                loss_func=loss_func,
                efficient_finetune=OmegaConf.select(config, "optimization.efficient_finetune"),
                mixup_fn=mixup_fn,
                mixup_off_epoch=OmegaConf.select(config, "data.mixup.turn_off_epoch"),
                model_postprocess_fn=model_postprocess_fn,
                trainable_param_names=trainable_param_names,
                **metrics_kwargs,
                **optimization_kwargs,
            )

        logger.debug(f"validation_metric_name: {task.validation_metric_name}")
        logger.debug(f"minmax_mode: {minmax_mode}")

        checkpoint_callback = AutoMMModelCheckpoint(
            dirpath=save_path,
            save_top_k=config.optimization.top_k,
            verbose=True,
            monitor=task.validation_metric_name,
            mode=minmax_mode,
            save_last=True,
        )
        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor=task.validation_metric_name,
            patience=config.optimization.patience,
            mode=minmax_mode,
        )
        lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
        model_summary = pl.callbacks.ModelSummary(max_depth=1)
        callbacks = [
            checkpoint_callback,
            early_stopping_callback,
            lr_callback,
            model_summary,
        ]

        use_ray_lightning = "_ray_lightning_plugin" in hpo_kwargs
        if hpo_mode:
            if use_ray_lightning:
                from ray_lightning.tune import TuneReportCheckpointCallback
            else:
                from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
            tune_report_callback = TuneReportCheckpointCallback(
                {f"{task.validation_metric_name}": f"{task.validation_metric_name}"},
                filename=RAY_TUNE_CHECKPOINT,
            )
            callbacks = [
                tune_report_callback,
                early_stopping_callback,
                lr_callback,
                model_summary,
            ]

        custom_checkpoint_plugin = AutoMMModelCheckpointIO(
            trainable_param_names=trainable_param_names, model_name_to_id=model.name_to_id
        )

        tb_logger = pl.loggers.TensorBoardLogger(
            save_dir=save_path,
            name="",
            version="",
        )

        num_gpus = compute_num_gpus(config_num_gpus=config.env.num_gpus, strategy=config.env.strategy)

        precision = infer_precision(num_gpus=num_gpus, precision=config.env.precision)

        if num_gpus == 0:  # CPU only training
            grad_steps = max(
                config.env.batch_size // (config.env.per_gpu_batch_size * config.env.num_nodes),
                1,
            )
        else:
            grad_steps = max(
                config.env.batch_size // (config.env.per_gpu_batch_size * num_gpus * config.env.num_nodes),
                1,
            )

        if not hpo_mode:
            if num_gpus <= 1:
                if config.env.strategy == DEEPSPEED_OFFLOADING:  # Offloading currently only tested for single GPU
                    assert version.parse(pl.__version__) >= version.parse(
                        DEEPSPEED_MIN_PL_VERSION
                    ), f"For DeepSpeed Offloading to work reliably you need at least pytorch-lightning version {DEEPSPEED_MIN_PL_VERSION}, however, found {pl.__version__}. Please update your pytorch-lightning version."
                    from .optimization.deepspeed import CustomDeepSpeedStrategy

                    strategy = CustomDeepSpeedStrategy(
                        stage=3,
                        offload_optimizer=True,
                        offload_parameters=False,
                        allgather_bucket_size=config.env.deepspeed_allgather_size,
                        reduce_bucket_size=config.env.deepspeed_allreduce_size,
                    )
                else:
                    strategy = None
            else:
                strategy = config.env.strategy
        else:
            # we don't support running each trial in parallel without ray lightning
            if use_ray_lightning:
                strategy = hpo_kwargs.get("_ray_lightning_plugin")
            else:
                strategy = None
                num_gpus = min(num_gpus, 1)

        config.env.num_gpus = num_gpus
        config.env.precision = precision
        config.env.strategy = strategy if not config.env.strategy == DEEPSPEED_OFFLOADING else DEEPSPEED_OFFLOADING
        self._config = config
        # save artifacts for the current running, except for model checkpoint, which will be saved in trainer
        self.save(save_path, standalone=standalone)

        blacklist_msgs = ["already configured with model summary"]
        log_filter = LogFilter(blacklist_msgs)
        with apply_log_filter(log_filter):
            trainer = pl.Trainer(
                accelerator="gpu" if num_gpus > 0 else None,
                devices=num_gpus
                if not use_ray_lightning and num_gpus > 0
                else None,  # ray lightning requires not specifying gpus
                auto_select_gpus=config.env.auto_select_gpus if num_gpus != 0 else False,
                num_nodes=config.env.num_nodes,
                precision=precision,
                strategy=strategy,
                benchmark=False,
                deterministic=config.env.deterministic,
                max_epochs=config.optimization.max_epochs,
                max_steps=config.optimization.max_steps,
                max_time=max_time,
                callbacks=callbacks,
                logger=tb_logger,
                gradient_clip_val=OmegaConf.select(config, "optimization.gradient_clip_val", default=1),
                gradient_clip_algorithm=OmegaConf.select(
                    config, "optimization.gradient_clip_algorithm", default="norm"
                ),
                accumulate_grad_batches=grad_steps,
                log_every_n_steps=OmegaConf.select(config, "optimization.log_every_n_steps", default=10),
                enable_progress_bar=enable_progress_bar,
                fast_dev_run=config.env.fast_dev_run,
                track_grad_norm=OmegaConf.select(config, "optimization.track_grad_norm", default=-1),
                val_check_interval=config.optimization.val_check_interval,
                check_val_every_n_epoch=config.optimization.check_val_every_n_epoch
                if hasattr(config.optimization, "check_val_every_n_epoch")
                else 1,
                reload_dataloaders_every_n_epochs=1,
                plugins=[custom_checkpoint_plugin],
            )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                ".*does not have many workers which may be a bottleneck. "
                "Consider increasing the value of the `num_workers` argument` "
                ".* in the `DataLoader` init to improve performance.*",
            )
            warnings.filterwarnings("ignore", "Checkpoint directory .* exists and is not empty.")
            trainer.fit(
                task,
                datamodule=train_dm,
                ckpt_path=ckpt_path if resume else None,  # this is to resume training that was broken accidentally
            )

        if trainer.global_rank == 0:
            # We do not perform averaging checkpoint in the case of hpo for each trial
            # We only averaging the checkpoint of the best trial in the end in the master process
            if not hpo_mode:
                self._top_k_average(
                    model=model,
                    save_path=save_path,
                    minmax_mode=minmax_mode,
                    is_distill=is_distill,
                    top_k_average_method=config.optimization.top_k_average_method,
                    val_df=val_df,
                    validation_metric_name=validation_metric_name,
                    strategy=strategy,
                    strict_loading=not trainable_param_names,  # Not strict loading if using parameter-efficient finetuning
                    standalone=standalone,
                )
        else:
            sys.exit(f"Training finished, exit the process with global_rank={trainer.global_rank}...")

    def _top_k_average(
        self,
        model,
        save_path,
        minmax_mode,
        is_distill,
        top_k_average_method,
        val_df,
        validation_metric_name,
        strategy=None,
        last_ckpt_path=None,
        strict_loading=True,
        standalone=True,
    ):
        best_k_models_yaml_path = os.path.join(save_path, BEST_K_MODELS_FILE)
        if os.path.exists(best_k_models_yaml_path):
            with open(best_k_models_yaml_path, "r") as f:
                best_k_models = yaml.safe_load(f)

        else:
            # In some cases, the training ends up too early (e.g., due to time_limit) so that there is
            # no saved best_k model checkpoints. In that scenario, we won't perform any model averaging.
            best_k_models = None
        if last_ckpt_path is None:
            last_ckpt_path = os.path.join(save_path, LAST_CHECKPOINT)

        if is_distill:
            prefix = "student_model."
        else:
            prefix = "model."

        if best_k_models:
            if top_k_average_method == UNIFORM_SOUP:
                logger.info(f"Start to fuse {len(best_k_models)} checkpoints via the uniform soup algorithm.")
                ingredients = top_k_model_paths = list(best_k_models.keys())
            else:
                top_k_model_paths = [
                    v[0]
                    for v in sorted(
                        list(best_k_models.items()),
                        key=lambda ele: ele[1],
                        reverse=(minmax_mode == MAX),
                    )
                ]
                if top_k_average_method == GREEDY_SOUP:
                    # Select the ingredients based on the methods proposed in paper
                    #  "Model soups: averaging weights of multiple fine-tuned models improves accuracy without
                    #  increasing inference time", https://arxiv.org/pdf/2203.05482.pdf
                    monitor_op = {MIN: operator.le, MAX: operator.ge}[minmax_mode]
                    ingredients = [top_k_model_paths[0]]
                    if len(top_k_model_paths) > 1:
                        logger.info(
                            f"Start to fuse {len(top_k_model_paths)} checkpoints via the greedy soup algorithm."
                        )

                        self._model = self._load_state_dict(
                            model=model,
                            path=top_k_model_paths[0],
                            prefix=prefix,
                            strict=strict_loading,
                        )
                        best_score = self.evaluate(val_df, [validation_metric_name])[validation_metric_name]
                        for i in range(1, len(top_k_model_paths)):
                            cand_avg_state_dict = average_checkpoints(
                                checkpoint_paths=ingredients + [top_k_model_paths[i]],
                            )
                            self._model = self._load_state_dict(
                                model=self._model,
                                state_dict=cand_avg_state_dict,
                                prefix=prefix,
                                strict=strict_loading,
                            )
                            cand_score = self.evaluate(val_df, [validation_metric_name])[validation_metric_name]
                            if monitor_op(cand_score, best_score):
                                # Add new ingredient
                                ingredients.append(top_k_model_paths[i])
                                best_score = cand_score
                elif top_k_average_method == BEST:
                    ingredients = [top_k_model_paths[0]]
                else:
                    raise ValueError(
                        f"The key for 'optimization.top_k_average_method' is not supported. "
                        f"We only support '{GREEDY_SOUP}', '{UNIFORM_SOUP}' and '{BEST}'. "
                        f"The provided value is '{top_k_average_method}'."
                    )
        else:
            # best_k_models is empty so we will manually save a checkpoint from the trainer
            # and use it as the main ingredients
            ingredients = [last_ckpt_path]
            top_k_model_paths = []
            # no checkpoints are available, do nothing
            if not os.path.isfile(last_ckpt_path):
                return

        # Average all the ingredients
        avg_state_dict = average_checkpoints(
            checkpoint_paths=ingredients,
        )
        self._model = self._load_state_dict(
            model=model,
            state_dict=avg_state_dict,
            prefix=prefix,
            strict=strict_loading,
        )

        if is_distill:
            avg_state_dict = self._replace_model_name_prefix(
                state_dict=avg_state_dict,
                old_prefix="student_model",
                new_prefix="model",
            )

        if not standalone:
            checkpoint = {"state_dict": avg_state_dict}
        else:
            if strategy and hasattr(strategy, "strategy_name") and strategy.strategy_name == DEEPSPEED_STRATEGY:
                checkpoint = {
                    "state_dict": {
                        name.partition("module.")[2]: param
                        for name, param in strategy.model._zero3_consolidated_16bit_state_dict().items()
                    }
                }
            else:
                checkpoint = {
                    "state_dict": {"model." + name: param for name, param in self._model.state_dict().items()}
                }

        torch.save(checkpoint, os.path.join(save_path, MODEL_CHECKPOINT))

        # clean old checkpoints + the intermediate files stored
        for per_path in top_k_model_paths:
            if os.path.isfile(per_path):
                os.remove(per_path)
        # remove the yaml file after cleaning the checkpoints
        if os.path.isfile(best_k_models_yaml_path):
            os.remove(best_k_models_yaml_path)
        # clean the last checkpoint
        if os.path.isfile(last_ckpt_path):
            os.remove(last_ckpt_path)

    def _default_predict(
        self,
        data: pd.DataFrame,
        df_preprocessor: MultiModalFeaturePreprocessor,
        data_processors: Dict,
        num_gpus: int,
        precision: Union[int, str],
        batch_size: int,
        strategy: str,
    ) -> List[Dict]:

        if self._config.env.strategy == DEEPSPEED_OFFLOADING and DEEPSPEED_MODULE not in sys.modules:
            # Need to initialize DeepSpeed and optimizer as currently required in Pytorch-Lighting integration of deepspeed.
            # TODO: Using optimiation_kwargs for inference is confusing and bad design. Remove as soon as fixed in pytorch-lighting.
            from .optimization.deepspeed import CustomDeepSpeedStrategy

            strategy = CustomDeepSpeedStrategy(
                stage=3,
                offload_optimizer=True,
                offload_parameters=False,
                allgather_bucket_size=self._config.env.deepspeed_allgather_size,
                reduce_bucket_size=self._config.env.deepspeed_allreduce_size,
            )
            norm_param_names = get_norm_layer_param_names(self._model)
            trainable_param_names = get_trainable_params_efficient_finetune(
                norm_param_names, efficient_finetune=OmegaConf.select(self._config, "optimization.efficient_finetune")
            )

            optimization_kwargs = dict(
                optim_type=self._config.optimization.optim_type,
                lr_choice=self._config.optimization.lr_choice,
                lr_schedule=self._config.optimization.lr_schedule,
                lr=self._config.optimization.learning_rate,
                lr_decay=self._config.optimization.lr_decay,
                end_lr=self._config.optimization.end_lr,
                lr_mult=self._config.optimization.lr_mult,
                weight_decay=self._config.optimization.weight_decay,
                warmup_steps=self._config.optimization.warmup_steps,
            )
        else:
            optimization_kwargs = {}
            trainable_param_names = []

        predict_dm = BaseDataModule(
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
            per_gpu_batch_size=batch_size,
            num_workers=self._config.env.num_workers_evaluation,
            predict_data=data,
        )

        if self._problem_type == NER:
            task = NerLitModule(
                model=self._model,
                model_postprocess_fn=self._model_postprocess_fn,
                efficient_finetune=OmegaConf.select(self._config, "optimization.efficient_finetune"),
                trainable_param_names=trainable_param_names,
                **optimization_kwargs,
            )
        elif self._pipeline == OBJECT_DETECTION:
            task = MMDetLitModule(
                model=self._model,
                **optimization_kwargs,
            )
        else:
            task = LitModule(
                model=self._model,
                model_postprocess_fn=self._model_postprocess_fn,
                efficient_finetune=OmegaConf.select(self._config, "optimization.efficient_finetune"),
                trainable_param_names=trainable_param_names,
                **optimization_kwargs,
            )

        blacklist_msgs = []
        if self._verbosity <= 3:  # turn off logging in prediction
            blacklist_msgs.append("Automatic Mixed Precision")
            blacklist_msgs.append("GPU available")
            blacklist_msgs.append("TPU available")
            blacklist_msgs.append("IPU available")
            blacklist_msgs.append("HPU available")
            blacklist_msgs.append("select gpus")
            blacklist_msgs.append("LOCAL_RANK")
        log_filter = LogFilter(blacklist_msgs)

        with apply_log_filter(log_filter):
            evaluator = pl.Trainer(
                accelerator="gpu" if num_gpus > 0 else None,
                devices=num_gpus if num_gpus > 0 else None,
                auto_select_gpus=self._config.env.auto_select_gpus if num_gpus != 0 else False,
                num_nodes=self._config.env.num_nodes,
                precision=precision,
                strategy=strategy,
                benchmark=False,
                enable_progress_bar=self._enable_progress_bar,
                deterministic=self._config.env.deterministic,
                max_epochs=-1,  # Add max_epochs to disable warning
                logger=False,
            )

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    ".*does not have many workers which may be a bottleneck. "
                    "Consider increasing the value of the `num_workers` argument` "
                    ".* in the `DataLoader` init to improve performance.*",
                )

                outputs = evaluator.predict(
                    task,
                    datamodule=predict_dm,
                )

        return outputs

    def _on_predict_start(
        self,
        config: DictConfig,
        data: Union[pd.DataFrame, dict, list],
        requires_label: bool,
    ):
        data = data_to_df(data=data)

        if self._column_types is None:
            allowable_dtypes, fallback_dtype = infer_dtypes_by_model_names(model_config=self._config.model)
            column_types = infer_column_types(
                data=data, allowable_column_types=allowable_dtypes, fallback_column_type=fallback_dtype
            )
            if self._label_column and self._label_column in data.columns:
                column_types = infer_label_column_type_by_problem_type_and_pipeline(
                    column_types=column_types,
                    label_columns=self._label_column,
                    problem_type=self._problem_type,
                    pipeline=self._pipeline,
                    data=data,
                )
            if self._pipeline == OBJECT_DETECTION:
                column_types = infer_rois_column_type(
                    column_types=column_types,
                    data=data,
                )
        else:  # called .fit() or .load()
            column_types = self._column_types

        if self._df_preprocessor is None:
            df_preprocessor = init_df_preprocessor(
                config=config,
                column_types=column_types,
                label_column=self._label_column,
                train_df_x=data,  # TODO: drop label like in line 884?
                train_df_y=data[self._label_column] if self._label_column else None,
            )
        else:  # called .fit() or .load()
            df_preprocessor = self._df_preprocessor

        data_processors = copy.deepcopy(self._data_processors)
        # For prediction data with no labels provided.
        if not requires_label:
            data_processors.pop(LABEL, None)

        return data, df_preprocessor, data_processors

    def evaluate_coco(
        self,
        anno_file_or_df: str,
        metrics: str,
    ):
        """
        Evaluate object detection model on a test dataset in COCO format.

        Parameters
        ----------
        anno_file
            The annotation file in COCO format
        """
        if isinstance(anno_file_or_df, str):
            anno_file = anno_file_or_df
            data = from_coco(anno_file)
        else:
            # during validation, it will call evaluate with df as input
            anno_file = self.detection_anno_train
            data = anno_file_or_df

        outputs = self._predict(
            data=data,
            requires_label=True,
        )  # outputs shape: num_batch, 1(["bbox"]), batch_size, 2(if using mask_rcnn)/na, 80, n, 5

        # Cache prediction results as COCO format # TODO: refactor this
        if not self._save_path:
            self._save_path = setup_outputdir(
                path=None,
                warn_if_exist=self._warn_if_exist,
            )
        self._save_path = os.path.abspath(os.path.expanduser(self._save_path))
        cache_path = os.path.join(self._save_path, "object_detection_result_cache.json")

        return cocoeval(
            outputs=outputs, data=data, anno_file=anno_file, cache_path=cache_path, metrics=metrics, tool="pycocotools"
        )

    def _process_batch(
        self,
        data: Union[pd.DataFrame, dict, list],
        df_preprocessor: MultiModalFeaturePreprocessor,
        data_processors: Dict,
    ):

        modality_features, modality_types, sample_num = apply_df_preprocessor(
            data=data,
            df_preprocessor=df_preprocessor,
            modalities=data_processors.keys(),
        )

        processed_features = []
        for i in range(sample_num):
            per_sample_features = get_per_sample_features(
                modality_features=modality_features,
                modality_types=modality_types,
                idx=i,
            )
            per_sample_features = apply_data_processor(
                per_sample_features=per_sample_features,
                data_processors=data_processors,
                feature_modalities=modality_types,
                is_training=False,
            )
            processed_features.append(per_sample_features)

        collate_fn = get_collate_fn(
            df_preprocessor=df_preprocessor, data_processors=data_processors, per_gpu_batch_size=sample_num
        )
        batch = collate_fn(processed_features)

        return batch

    def _realtime_predict(
        self,
        data: pd.DataFrame,
        df_preprocessor: MultiModalFeaturePreprocessor,
        data_processors: Dict,
        num_gpus: int,
        precision: Union[int, str],
    ) -> List[Dict]:
        batch = self._process_batch(
            data=data,
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
        )
        output = infer_batch(
            batch=batch,
            model=self._model,
            precision=precision,
            num_gpus=num_gpus,
            model_postprocess_fn=self._model_postprocess_fn,
        )
        return [output]

    def _predict(
        self,
        data: Union[pd.DataFrame, dict, list],
        requires_label: bool,
        realtime: Optional[bool] = None,
        seed: Optional[int] = 123,
    ) -> List[Dict]:

        with apply_log_filter(LogFilter("Global seed set to")):  # Ignore the log "Global seed set to"
            pl.seed_everything(seed, workers=True)

        data, df_preprocessor, data_processors = self._on_predict_start(
            config=self._config,
            data=data,
            requires_label=requires_label,
        )

        strategy = "dp"  # default used in inference.

        num_gpus = compute_num_gpus(config_num_gpus=self._config.env.num_gpus, strategy=strategy)

        if self._pipeline == OBJECT_DETECTION:
            # strategy = "ddp" # TODO: enable multigpu inference
            num_gpus = 1

        if num_gpus == 1:
            strategy = None

        precision = infer_precision(num_gpus=num_gpus, precision=self._config.env.precision, cpu_only_warning=False)

        if not realtime:
            batch_size = compute_inference_batch_size(
                per_gpu_batch_size=self._config.env.per_gpu_batch_size,
                eval_batch_size_ratio=OmegaConf.select(self._config, "env.eval_batch_size_ratio"),
                per_gpu_batch_size_evaluation=self._config.env.per_gpu_batch_size_evaluation,  # backward compatibility.
                num_gpus=num_gpus,
                strategy=strategy,
            )

        if realtime is None:
            realtime = use_realtime(data=data, data_processors=data_processors, batch_size=batch_size)

        if realtime:
            outputs = self._realtime_predict(
                data=data,
                df_preprocessor=df_preprocessor,
                data_processors=data_processors,
                num_gpus=num_gpus,
                precision=precision,
            )
        else:
            outputs = self._default_predict(
                data=data,
                df_preprocessor=df_preprocessor,
                data_processors=data_processors,
                num_gpus=num_gpus,
                precision=precision,
                batch_size=batch_size,
                strategy=strategy,
            )

        return outputs

    def evaluate(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        metrics: Optional[Union[str, List[str]]] = None,
        return_pred: Optional[bool] = False,
        realtime: Optional[bool] = None,
        seed: Optional[int] = 123,
    ):
        """
        Evaluate model on a test dataset.

        Parameters
        ----------
        data
            A dataframe, containing the same columns as the training data.
            Or a str, that is a path of the annotation file for detection.
        metrics
            A list of metric names to report.
            If None, we only return the score for the stored `_eval_metric_name`.
        return_pred
            Whether to return the prediction result of each row.
        realtime
            Whether to do realtime inference, which is efficient for small data (default None).
            If not specified, we would infer it on based on the data modalities
            and sample number.

        Returns
        -------
        A dictionary with the metric names and their corresponding scores.
        Optionally return a dataframe of prediction results.
        """
        if self._pipeline == OBJECT_DETECTION:
            return self.evaluate_coco(data, metrics)

        if self._problem_type == NER:
            ret_type = NER_RET
        else:
            ret_type = LOGITS

        outputs = self._predict(
            data=data,
            requires_label=True,
            realtime=realtime,
            seed=seed,
        )
        logits = extract_from_output(ret_type=ret_type, outputs=outputs)

        metric_data = {}
        if self._problem_type in [BINARY, MULTICLASS]:
            y_pred_prob = logits_to_prob(logits)
            metric_data[Y_PRED_PROB] = y_pred_prob

        y_pred = self._df_preprocessor.transform_prediction(
            y_pred=logits,
            inverse_categorical=False,
        )
        y_pred_inv = self._df_preprocessor.transform_prediction(
            y_pred=logits,
            inverse_categorical=True,
        )

        if self._problem_type == NER:
            y_true = self._df_preprocessor.transform_label_for_metric(df=data, tokenizer=self._model.tokenizer)
        else:
            y_true = self._df_preprocessor.transform_label_for_metric(df=data)

        metric_data.update(
            {
                Y_PRED: y_pred,
                Y_TRUE: y_true,
            }
        )

        metrics_is_none = False

        if metrics is None:
            metrics_is_none = True
            metrics = [self._eval_metric_name]
        if isinstance(metrics, str):
            metrics = [metrics]

        results = {}
        if self._problem_type == NER:
            score = compute_score(
                metric_data=metric_data,
                metric_name=self._eval_metric_name.lower(),
            )
            score = {k.lower(): v for k, v in score.items()}
            if metrics_is_none:
                results = score
            else:
                for per_metric in metrics:
                    if per_metric.lower() in score:
                        results.update({per_metric: score[per_metric.lower()]})
                    else:
                        logger.warning(f"Warning: {per_metric} is not a suppported evaluation metric!")
                if not results:
                    results = score  # If the results dict is empty, return all scores.
        else:
            for per_metric in metrics:
                pos_label = try_to_infer_pos_label(
                    data_config=self._config.data,
                    label_encoder=self._df_preprocessor.label_generator,
                    problem_type=self._problem_type,
                )
                score = compute_score(
                    metric_data=metric_data,
                    metric_name=per_metric.lower(),
                    pos_label=pos_label,
                )
                results[per_metric] = score

        if return_pred:
            return results, self._as_pandas(data=data, to_be_converted=y_pred_inv)
        else:
            return results

    def _match_queries_and_candidates(
        self,
        query_data: Union[pd.DataFrame, dict, list],
        candidate_data: Union[pd.DataFrame, dict, list],
        return_prob: Optional[bool] = False,
    ):
        query_embeddings = self.extract_embedding(query_data, as_tensor=True)
        assert (
            len(query_embeddings) == 1
        ), f"Multiple embedding types `{query_embeddings.keys()}` exist in query data. Please reduce them to one type."
        query_embeddings = list(query_embeddings.values())[0]

        candidate_embeddings = self.extract_embedding(candidate_data, as_tensor=True)
        assert (
            len(candidate_embeddings) == 1
        ), f"Multiple embedding types `{candidate_embeddings.keys()}` exist in candidate data. Please reduce them to one type."
        candidate_embeddings = list(candidate_embeddings.values())[0]

        if return_prob:
            ret = (100.0 * query_embeddings @ candidate_embeddings.T).float().softmax(dim=-1)
        else:
            ret = (query_embeddings @ candidate_embeddings.T).argmax(dim=-1)

        ret = tensor_to_ndarray(ret)

        return ret

    def predict(
        self,
        data: Union[pd.DataFrame, dict, list],
        candidate_data: Optional[Union[pd.DataFrame, dict, list]] = None,
        as_pandas: Optional[bool] = None,
        realtime: Optional[bool] = None,
        seed: Optional[int] = 123,
    ):
        """
        Predict values for the label column of new data.

        Parameters
        ----------
        data
             The data to make predictions for. Should contain same column names as training data and
              follow same format (except for the `label` column).
        candidate_data
            The candidate data from which to search the query data's matches.
        as_pandas
            Whether to return the output as a pandas DataFrame(Series) (True) or numpy array (False).
        realtime
            Whether to do realtime inference, which is efficient for small data (default None).
            If not specified, we would infer it on based on the data modalities
            and sample number.

        Returns
        -------
        Array of predictions, one corresponding to each row in given dataset.
        """
        if self._pipeline == OBJECT_DETECTION or self._pipeline == OCR_TEXT_DETECTION:
            ret_type = BBOX
        elif self._pipeline == OCR_TEXT_RECOGNITION:
            ret_type = [TEXT, SCORE]
        else:
            ret_type = LOGITS

        if self._problem_type == NER:
            ret_type = NER_RET

        if candidate_data:
            pred = self._match_queries_and_candidates(
                query_data=data,
                candidate_data=candidate_data,
                return_prob=False,
            )
        else:
            outputs = self._predict(
                data=data,
                requires_label=False,
                realtime=realtime,
                seed=seed,
            )

            if self._pipeline == OCR_TEXT_RECOGNITION:
                logits = []
                for r_type in ret_type:
                    logits.append(extract_from_output(outputs=outputs, ret_type=r_type))
            else:
                logits = extract_from_output(outputs=outputs, ret_type=ret_type)

            if self._df_preprocessor:
                pred = self._df_preprocessor.transform_prediction(
                    y_pred=logits,
                )
            else:
                if isinstance(logits, (torch.Tensor, np.ndarray)) and logits.ndim == 2:
                    pred = logits.argmax(axis=1)
                else:
                    pred = logits

        if (as_pandas is None and isinstance(data, pd.DataFrame)) or as_pandas is True:
            pred = self._as_pandas(data=data, to_be_converted=pred)

        return pred

    def predict_proba(
        self,
        data: Union[pd.DataFrame, dict, list],
        candidate_data: Optional[Union[pd.DataFrame, dict, list]] = None,
        as_pandas: Optional[bool] = None,
        as_multiclass: Optional[bool] = True,
        realtime: Optional[bool] = None,
        seed: Optional[int] = 123,
    ):
        """
        Predict probabilities class probabilities rather than class labels.
        This is only for the classification tasks. Calling it for a regression task will throw an exception.

        Parameters
        ----------
        data
            The data to make predictions for. Should contain same column names as training data and
              follow same format (except for the `label` column).
        candidate_data
            The candidate data from which to search the query data's matches.
        as_pandas
            Whether to return the output as a pandas DataFrame(Series) (True) or numpy array (False).
        as_multiclass
            Whether to return the probability of all labels or
            just return the probability of the positive class for binary classification problems.
        realtime
            Whether to do realtime inference, which is efficient for small data (default None).
            If not specified, we would infer it on based on the data modalities
            and sample number.

        Returns
        -------
        Array of predicted class-probabilities, corresponding to each row in the given data.
        When as_multiclass is True, the output will always have shape (#samples, #classes).
        Otherwise, the output will have shape (#samples,)
        """
        assert self._problem_type not in [
            REGRESSION,
            NER,
        ], f"Problem {self._problem_type} has no probability output."

        if candidate_data:
            prob = self._match_queries_and_candidates(
                query_data=data,
                candidate_data=candidate_data,
                return_prob=True,
            )
        else:
            outputs = self._predict(
                data=data,
                requires_label=False,
                realtime=realtime,
                seed=seed,
            )
            logits = extract_from_output(outputs=outputs, ret_type=LOGITS)

            prob = logits_to_prob(logits)

        if not as_multiclass:
            if self._problem_type == BINARY:
                pos_label = try_to_infer_pos_label(
                    data_config=self._config.data,
                    label_encoder=self._df_preprocessor.label_generator,
                    problem_type=self._problem_type,
                )
                prob = prob[:, pos_label]

        if (as_pandas is None and isinstance(data, pd.DataFrame)) or as_pandas is True:
            prob = self._as_pandas(data=data, to_be_converted=prob)

        return prob

    def extract_embedding(
        self,
        data: Union[pd.DataFrame, dict, list],
        return_masks: Optional[bool] = False,
        as_tensor: Optional[bool] = False,
        as_pandas: Optional[bool] = False,
        realtime: Optional[bool] = None,
    ):
        """
        Extract features for each sample, i.e., one row in the provided dataframe `data`.

        Parameters
        ----------
        data
            The data to extract embeddings for. Should contain same column names as training dataset and
            follow same format (except for the `label` column).
        return_masks
            If true, returns a mask dictionary, whose keys are the same as those in the features dictionary.
            If a sample has empty input in feature column `image_0`, the sample will has mask 0 under key `image_0`.
        as_tensor
            Whether to return a Pytorch tensor.
        as_pandas
            Whether to return the output as a pandas DataFrame (True) or numpy array (False).
        realtime
            Whether to do realtime inference, which is efficient for small data (default None).
            If not specified, we would infer it on based on the data modalities
            and sample number.

        Returns
        -------
        Array of embeddings, corresponding to each row in the given data.
        It will have shape (#samples, D) where the embedding dimension D is determined
        by the neural network's architecture.
        """
        turn_on_off_feature_column_info(
            data_processors=self._data_processors,
            flag=True,
        )
        outputs = self._predict(
            data=data,
            requires_label=False,
            realtime=realtime,
        )
        if self._pipeline in [FEATURE_EXTRACTION, ZERO_SHOT_IMAGE_CLASSIFICATION]:
            features = extract_from_output(outputs=outputs, ret_type=COLUMN_FEATURES, as_ndarray=as_tensor is False)
            if return_masks:
                masks = extract_from_output(outputs=outputs, ret_type=MASKS, as_ndarray=as_tensor is False)
        else:
            features = extract_from_output(outputs=outputs, ret_type=FEATURES, as_ndarray=as_tensor is False)

        if as_pandas:
            features = pd.DataFrame(features, index=data.index)
            if return_masks:
                masks = pd.DataFrame(masks, index=data.index)

        if return_masks:
            return features, masks
        else:
            return features

    def _as_pandas(
        self,
        data: Union[pd.DataFrame, dict, list],
        to_be_converted: Union[np.ndarray, dict],
    ):
        if isinstance(data, pd.DataFrame):
            index = data.index
        else:
            index = None
        if isinstance(to_be_converted, list) or (
            isinstance(to_be_converted, np.ndarray) and to_be_converted.ndim == 1
        ):
            return pd.Series(to_be_converted, index=index, name=self._label_column)
        else:
            return pd.DataFrame(to_be_converted, index=index, columns=self.class_labels)

    @staticmethod
    def _load_state_dict(
        model: nn.Module,
        state_dict: dict = None,
        path: str = None,
        prefix: str = "model.",
        strict: bool = True,
    ):
        if state_dict is None:
            state_dict = torch.load(path, map_location=torch.device("cpu"))["state_dict"]
        state_dict = {k.partition(prefix)[2]: v for k, v in state_dict.items() if k.startswith(prefix)}
        load_result = model.load_state_dict(state_dict, strict=strict)
        assert (
            len(load_result.unexpected_keys) == 0
        ), f"Load model failed, unexpected keys {load_result.unexpected_keys.__str__()}"
        return model

    @staticmethod
    def _replace_model_name_prefix(
        state_dict: dict,
        old_prefix: str,
        new_prefix: str,
    ):
        start_idx = len(old_prefix)
        state_dict_processed = {
            new_prefix + k[start_idx:]: v for k, v in state_dict.items() if k.startswith(old_prefix)
        }
        return state_dict_processed

    def save(self, path: str, standalone: Optional[bool] = True):
        """
        Save this predictor to file in directory specified by `path`.

        Parameters
        ----------
        path
            The directory to save this predictor.
        standalone
            Whether to save the downloaded model for offline deployment.
            When standalone = True, save the transformers.CLIPModel and transformers.AutoModel to os.path.join(path,model_name),
            and reset the associate model.model_name.checkpoint_name start with `local://` in config.yaml.
            When standalone = False, the saved artifact may require an online environment to process in load().
        """

        config = copy.deepcopy(self._config)
        if standalone and (
            not OmegaConf.select(config, "optimization.efficient_finetune")
            or OmegaConf.select(config, "optimization.efficient_finetune") == "None"
        ):
            config = save_pretrained_model_configs(model=self._model, config=config, path=path)

        os.makedirs(path, exist_ok=True)
        OmegaConf.save(config=config, f=os.path.join(path, "config.yaml"))

        with open(os.path.join(path, "df_preprocessor.pkl"), "wb") as fp:
            pickle.dump(self._df_preprocessor, fp)

        # Save text tokenizers before saving data processors
        data_processors = copy.deepcopy(self._data_processors)
        if TEXT in data_processors:
            data_processors[TEXT] = save_text_tokenizers(
                text_processors=data_processors[TEXT],
                path=path,
            )

        with open(os.path.join(path, "data_processors.pkl"), "wb") as fp:
            pickle.dump(data_processors, fp)

        with open(os.path.join(path, f"assets.json"), "w") as fp:
            json.dump(
                {
                    "column_types": self._column_types,
                    "label_column": self._label_column,
                    "problem_type": self._problem_type,
                    "pipeline": self._pipeline,
                    "eval_metric_name": self._eval_metric_name,
                    "validation_metric_name": self._validation_metric_name,
                    "output_shape": self._output_shape,
                    "save_path": self._save_path,
                    "pretrained_path": self._pretrained_path,
                    "version": ag_version.__version__,
                },
                fp,
                ensure_ascii=True,
            )

        # In case that users save to a path, which is not the original save_path.
        if os.path.abspath(path) != os.path.abspath(self._save_path):
            model_path = os.path.join(self._save_path, "model.ckpt")
            if os.path.isfile(model_path):
                shutil.copy(model_path, path)
            else:
                # FIXME(?) Fix the saving logic
                RuntimeError(
                    f"Cannot find the model checkpoint in '{model_path}'. Have you removed the folder that "
                    f"is created in .fit()? Currently, .save() won't function appropriately if that folder is "
                    f"removed."
                )

    def export_onnx(
        self,
        onnx_path: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        batch_size: Optional[int] = None,
        verbose: Optional[bool] = False,
        opset_version: Optional[int] = 13,
    ):
        """
        Export this predictor's model to ONNX file.

        Parameters
        ----------
        onnx_path
            The export path of onnx model.
        data
            Raw data used to trace and export the model.
            If this is None, will check if a processed batch is provided.
        batch_size
            The batch_size of export model's input.
            Normally the batch_size is a dynamic axis, so we could use a small value for faster export.
        verbose
            verbose flag in torch.onnx.export.
        opset_version
            opset_version flag in torch.onnx.export.
        """
        # TODO: Support CLIP
        # TODO: Add test

        valid_input, dynamic_axes, default_onnx_path, batch = get_onnx_input(
            pipeline=self._pipeline, config=self._config
        )

        if not batch_size:
            batch_size = 2  # batch_size should be a dynamic_axis, so we could use a small value for faster export
        if data is not None:
            batch = self.get_processed_batch_for_deployment(
                data=data, valid_input=valid_input, onnx_tracing=True, batch_size=batch_size
            )

        if not onnx_path:
            onnx_path = default_onnx_path

        torch.onnx.export(
            self._model.eval(),
            batch,
            onnx_path,
            opset_version=opset_version,
            verbose=verbose,
            input_names=valid_input,
            dynamic_axes=dynamic_axes,
        )

    def get_processed_batch_for_deployment(
        self,
        data: pd.DataFrame,
        valid_input: Optional[List] = None,
        onnx_tracing: bool = False,
        batch_size: int = None,
        to_numpy: bool = True,
        requires_label: bool = False,
    ):
        """
        Get the processed batch of raw data given.

        Parameters
        ----------
        data
            The raw data to process
        valid_input
            Used to filter valid data. No filter happens if it is empty.
        onnx_tracing
            If the output is used for onnx tracing.
        batch_size
            The batch_size of output batch.
            If onnx_tracing, it will only output one mini-batch, and all int tensor values will be converted to long.
        to_numpy
            Output numpy array if True. Only valid if not onnx_tracing.

        Returns
        -------
        Tensor or numpy array.
        The output processed batch could be used for export/evaluate deployed model.
        """
        # TODO: add support for data = dict or list
        if onnx_tracing:
            if batch_size:
                data = data[:batch_size]
            else:
                data = data[:2]

        data, df_preprocessor, data_processors = self._on_predict_start(
            config=self._config,
            data=data,
            requires_label=requires_label,
        )

        batch = self._process_batch(
            data=data,
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
        )

        ret = {}
        for k in batch:
            if valid_input and k not in valid_input:
                continue
            if onnx_tracing:
                ret[k] = batch[k].long() if isinstance(batch[k], torch.IntTensor) else batch[k]
            elif to_numpy:
                ret[k] = batch[k].cpu().detach().numpy().astype(int)
            else:
                ret[k] = batch[k]
        if not onnx_tracing:
            if batch_size:
                raise NotImplementedError("We should split the batch here.")  # TODO
        return ret

    @staticmethod
    def _load_metadata(
        predictor: MultiModalPredictor,
        path: str,
        resume: Optional[bool] = False,
        verbosity: Optional[int] = 3,
    ):
        path = os.path.abspath(os.path.expanduser(path))
        assert os.path.isdir(path), f"'{path}' must be an existing directory."
        config = OmegaConf.load(os.path.join(path, "config.yaml"))

        config = get_local_pretrained_config_paths(
            config=config, path=path
        )  # check the config to load offline pretrained model configs

        with open(os.path.join(path, "assets.json"), "r") as fp:
            assets = json.load(fp)

        with open(os.path.join(path, "df_preprocessor.pkl"), "rb") as fp:
            df_preprocessor = CustomUnpickler(fp).load()

        try:
            with open(os.path.join(path, "data_processors.pkl"), "rb") as fp:
                data_processors = CustomUnpickler(fp).load()
            # Load text tokenizers after loading data processors.
            if TEXT in data_processors:
                data_processors[TEXT] = load_text_tokenizers(
                    text_processors=data_processors[TEXT],
                    path=path,
                )
            # backward compatibility. Add feature column names in each data processor.
            data_processors = assign_feature_column_names(
                data_processors=data_processors,
                df_preprocessor=df_preprocessor,
            )

            # Only keep the modalities with non-empty processors.
            data_processors = {k: v for k, v in data_processors.items() if len(v) > 0}
        except:  # backward compatibility. reconstruct the data processor in case something went wrong.
            data_processors = None

        predictor._label_column = assets["label_column"]
        predictor._problem_type = assets["problem_type"]
        if "pipeline" in assets:  # back compatibility
            predictor._pipeline = assets["pipeline"]
        predictor._eval_metric_name = assets["eval_metric_name"]
        predictor._verbosity = verbosity
        predictor._resume = resume
        predictor._save_path = path  # in case the original exp dir is copied to somewhere else
        predictor._pretrain_path = path
        predictor._config = config
        predictor._output_shape = assets["output_shape"]
        predictor._column_types = assets["column_types"]
        predictor._validation_metric_name = assets["validation_metric_name"]
        predictor._df_preprocessor = df_preprocessor
        predictor._data_processors = data_processors

        return predictor

    @classmethod
    def load(
        cls,
        path: str,
        resume: Optional[bool] = False,
        verbosity: Optional[int] = 3,
    ):
        """
        Load a predictor object from a directory specified by `path`. The to-be-loaded predictor
        can be completely or partially trained by .fit(). If a previous training has completed,
        it will load the checkpoint `model.ckpt`. Otherwise if a previous training accidentally
        collapses in the middle, it can load the `last.ckpt` checkpoint by setting `resume=True`.

        Parameters
        ----------
        path
            The directory to load the predictor object.
        resume
            Whether to resume training from `last.ckpt`. This is useful when a training was accidentally
            broken during the middle and we want to resume the training from the last saved checkpoint.
        verbosity
            Verbosity levels range from 0 to 4 and control how much information is printed.
            Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).

        Returns
        -------
        The loaded predictor object.
        """
        path = os.path.abspath(os.path.expanduser(path))
        assert os.path.isdir(path), f"'{path}' must be an existing directory."
        predictor = cls(label="dummy_label")
        predictor = cls._load_metadata(predictor=predictor, path=path, resume=resume, verbosity=verbosity)

        efficient_finetune = OmegaConf.select(predictor._config, "optimization.efficient_finetune")

        model = create_fusion_model(
            config=predictor._config,
            num_classes=predictor._output_shape,
            num_numerical_columns=len(predictor._df_preprocessor.numerical_feature_names),
            num_categories=predictor._df_preprocessor.categorical_num_categories,
            pretrained=False
            if not efficient_finetune or efficient_finetune == "None"
            else True,  # set "pretrain=False" to prevent downloading online models
        )

        if predictor._data_processors is None:
            predictor._data_processors = create_fusion_data_processors(
                config=predictor._config,
                model=model,
            )

        resume_ckpt_path = os.path.join(path, LAST_CHECKPOINT)
        final_ckpt_path = os.path.join(path, MODEL_CHECKPOINT)
        if resume:  # resume training which crashed before
            if not os.path.isfile(resume_ckpt_path):
                if os.path.isfile(final_ckpt_path):
                    raise ValueError(
                        f"Resuming checkpoint '{resume_ckpt_path}' doesn't exist, but "
                        f"final checkpoint '{final_ckpt_path}' exists, which means training "
                        f"is already completed."
                    )
                else:
                    raise ValueError(
                        f"Resuming checkpoint '{resume_ckpt_path}' and "
                        f"final checkpoint '{final_ckpt_path}' both don't exist. "
                        f"Consider starting training from scratch."
                    )
            load_path = resume_ckpt_path
            logger.info(f"Resume training from checkpoint: '{resume_ckpt_path}'")
            ckpt_path = resume_ckpt_path
        else:  # load a model checkpoint for prediction, evaluation, or continuing training on new data
            if not os.path.isfile(final_ckpt_path):
                if os.path.isfile(resume_ckpt_path):
                    raise ValueError(
                        f"Final checkpoint '{final_ckpt_path}' doesn't exist, but "
                        f"resuming checkpoint '{resume_ckpt_path}' exists, which means training "
                        f"is not done yet. Consider resume training from '{resume_ckpt_path}'."
                    )
                else:
                    raise ValueError(
                        f"Resuming checkpoint '{resume_ckpt_path}' and "
                        f"final checkpoint '{final_ckpt_path}' both don't exist. "
                        f"Consider starting training from scratch."
                    )
            load_path = final_ckpt_path
            logger.info(f"Load pretrained checkpoint: {os.path.join(path, MODEL_CHECKPOINT)}")
            ckpt_path = None  # must set None since we do not resume training

        model = cls._load_state_dict(
            model=model,
            path=load_path,
            strict=not efficient_finetune or efficient_finetune == "None",
        )

        predictor._ckpt_path = ckpt_path
        predictor._model = model
        if not resume:
            predictor._continuous_training = True

        loss_func = get_loss_func(
            problem_type=predictor._problem_type,
            mixup_active=False,
            loss_func_name=OmegaConf.select(predictor._config, "optimization.loss_function"),
        )

        model_postprocess_fn = get_model_postprocess_fn(
            problem_type=predictor._problem_type,
            loss_func=loss_func,
        )
        predictor._model_postprocess_fn = model_postprocess_fn

        return predictor

    @property
    def class_labels(self):
        """
        The original name of the class labels.
        For example, the tabular data may contain classes equal to
        "entailment", "contradiction", "neutral". Internally, these will be converted to
        0, 1, 2, ...
        This function returns the original names of these raw labels.

        Returns
        -------
        List that contain the class names. It will be None if it's not a classification problem.
        """
        if self._problem_type == MULTICLASS or self._problem_type == BINARY:
            return self._df_preprocessor.label_generator.classes_
        else:
            warnings.warn("Accessing class names for a non-classification problem. Return None.")
            return None

    @property
    def positive_class(self):
        """
        Name of the class label that will be mapped to 1.
        This is only meaningful for binary classification problems.

        It is useful for computing metrics such as F1 which require a positive and negative class.
        You may refer to https://en.wikipedia.org/wiki/F-score for more details.
        In binary classification, :class:`TextPredictor.predict_proba(as_multiclass=False)`
        returns the estimated probability that each row belongs to the positive class.
        Will print a warning and return None if called when `predictor.problem_type != 'binary'`.

        Returns
        -------
        The positive class name in binary classification or None if the problem is not binary classification.
        """
        if self.problem_type != BINARY:
            logger.warning(
                f"Warning: Attempted to retrieve positive class label in a non-binary problem. "
                f"Positive class labels only exist in binary classification. "
                f"Returning None instead. self.problem_type is '{self.problem_type}'"
                f" but positive_class only exists for '{BINARY}'."
            )
            return None
        else:
            return self.class_labels[1]


class AutoMMPredictor(MultiModalPredictor):
    def __init__(self, **kwargs):
        warnings.warn(
            "AutoMMPredictor has been renamed as 'MultiModalPredictor'. "
            "Consider to use MultiModalPredictor instead. Using AutoMMPredictor will "
            "raise an exception starting in v0.7."
        )
        super(AutoMMPredictor, self).__init__(**kwargs)
