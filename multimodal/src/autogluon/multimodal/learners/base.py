from __future__ import annotations

import copy
import json
import logging
import operator
import os
import pickle
import shutil
import sys
import time
import warnings
from datetime import timedelta
from typing import Callable, Dict, List, Optional, Union

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import yaml
from lightning.pytorch.strategies import DeepSpeedStrategy
from omegaconf import DictConfig, OmegaConf
from packaging import version
from torch import nn

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.common.utils.try_import import try_import_ray
from autogluon.core.metrics import Scorer, get_metric
from autogluon.core.utils.loaders import load_pd

from .. import version as ag_version
from ..constants import (
    BEST,
    BEST_K_MODELS_FILE,
    BINARY,
    COLUMN_FEATURES,
    DDP,
    DEEPSPEED_MIN_PL_VERSION,
    DEEPSPEED_MODULE,
    DEEPSPEED_OFFLOADING,
    DISTILLER,
    DOCUMENT,
    FEATURE_EXTRACTION,
    FEATURES,
    FEW_SHOT,
    FEW_SHOT_CLASSIFICATION,
    GREEDY_SOUP,
    IMAGE_BASE64_STR,
    IMAGE_BYTEARRAY,
    IMAGE_PATH,
    LABEL,
    LAST_CHECKPOINT,
    LOGITS,
    MASKS,
    MAX,
    MIN,
    MODEL_CHECKPOINT,
    MULTICLASS,
    NER,
    RAY_TUNE_CHECKPOINT,
    REGRESSION,
    TEXT,
    TEXT_NER,
    TORCH_COMPILE_MIN_VERSION,
    UNIFORM_SOUP,
    Y_PRED,
    Y_PRED_PROB,
    Y_TRUE,
    ZERO_SHOT_IMAGE_CLASSIFICATION,
)
from ..data import (
    BaseDataModule,
    MultiModalFeaturePreprocessor,
    create_fusion_data_processors,
    data_to_df,
    get_mixup,
    infer_column_types,
    infer_dtypes_by_model_names,
    infer_output_shape,
    infer_problem_type,
    infer_scarcity_mode_by_data_size,
    init_df_preprocessor,
    is_image_column,
    split_train_tuning_data,
    turn_on_off_feature_column_info,
)
from ..models import (
    create_fusion_model,
    get_model_postprocess_fn,
    is_lazy_weight_tensor,
    list_timm_models,
    select_model,
)
from ..optim import (
    compute_score,
    get_aug_loss_func,
    get_loss_func,
    get_minmax_mode,
    get_norm_layer_param_names,
    get_peft_param_names,
    get_stopping_threshold,
    get_torchmetric,
    infer_metrics,
)
from ..optim.lit_distiller import DistillerLitModule
from ..optim.lit_module import LitModule
from ..utils import (
    AutoMMModelCheckpoint,
    AutoMMModelCheckpointIO,
    DDPPredictionWriter,
    DistillationMixin,
    ExportMixin,
    LogFilter,
    RealtimeMixin,
    apply_log_filter,
    average_checkpoints,
    compute_inference_batch_size,
    compute_num_gpus,
    extract_from_output,
    filter_hyperparameters,
    get_config,
    get_dir_ckpt_paths,
    get_gpu_message,
    get_load_ckpt_paths,
    get_local_pretrained_config_paths,
    hyperparameter_tune,
    infer_precision,
    infer_problem_type_by_eval_metric,
    is_interactive_env,
    is_interactive_strategy,
    logits_to_prob,
    on_fit_end_message,
    on_fit_per_run_start_message,
    on_fit_start_message,
    run_ddp_only_once,
    save_pretrained_model_configs,
    setup_save_path,
    split_hyperparameters,
    tensor_to_ndarray,
    update_config_by_rules,
    update_hyperparameters,
    update_tabular_config_by_resources,
)
from ..utils.problem_types import PROBLEM_TYPES_REG

pl_logger = logging.getLogger("lightning")
pl_logger.propagate = False  # https://github.com/Lightning-AI/lightning/issues/4621
logger = logging.getLogger(__name__)


class BaseLearner(ExportMixin, DistillationMixin, RealtimeMixin):
    """ """

    def __init__(
        self,
        label: Optional[str] = None,
        problem_type: Optional[str] = None,
        presets: Optional[str] = None,
        eval_metric: Optional[Union[str, Scorer]] = None,
        hyperparameters: Optional[dict] = None,
        path: Optional[str] = None,
        verbosity: Optional[int] = 2,
        warn_if_exist: Optional[bool] = True,
        enable_progress_bar: Optional[bool] = None,
        pretrained: Optional[bool] = True,
        validation_metric: Optional[str] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        label
            Name of the column that contains the target variable to predict.
        problem_type
            Type of the prediction problem. We support standard problems like

            - 'binary': Binary classification
            - 'multiclass': Multi-class classification
            - 'regression': Regression
            - 'classification': Classification problems include 'binary' and 'multiclass' classification.

            In addition, we support advanced problems such as

            - 'object_detection': Object detection
            - 'ner' or 'named_entity_recognition': Named entity extraction
            - 'text_similarity': Text-text similarity problem
            - 'image_similarity': Image-image similarity problem
            - 'image_text_similarity': Text-image similarity problem
            - 'feature_extraction': Extracting feature (only support inference)
            - 'zero_shot_image_classification': Zero-shot image classification (only support inference)
            - 'few_shot_classification': Few-shot classification for image or text data.

            For certain problem types, the default behavior is to load a pretrained model based on
            the presets / hyperparameters and the learner will support zero-shot inference
            (running inference without .fit()). This includes the following
            problem types:

            - 'object_detection'
            - 'text_similarity'
            - 'image_similarity'
            - 'image_text_similarity'
            - 'feature_extraction'
            - 'zero_shot_image_classification'
            - 'few_shot_classification'
        presets
            Presets regarding model quality, e.g., best_quality, high_quality, and medium_quality.
        eval_metric
            Evaluation metric name. If `eval_metric = None`, it is automatically chosen based on `problem_type`.
            Defaults to 'accuracy' for multiclass classification, `roc_auc` for binary classification, and 'root_mean_squared_error' for regression.
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
        pretrained
            Whether to init model with pretrained weights. If False, it creates a model with random initialization.
        validation_metric
            Validation metric name. If `validation_metric = None`, it is automatically chosen based on `problem_type`.
            Defaults to 'accuracy' for multiclass classification, `roc_auc` for binary classification, and 'root_mean_squared_error' for regression.
        """
        self._eval_metric_name = None
        self._eval_metric_func = None
        if isinstance(eval_metric, str):
            self._eval_metric_name = eval_metric.lower()
            self.set_eval_metric_func()
        elif isinstance(eval_metric, Scorer):
            self._eval_metric_name = eval_metric.name
            self._eval_metric_func = eval_metric

        self._problem_type = infer_problem_type_by_eval_metric(
            eval_metric_name=self._eval_metric_name,
            problem_type=problem_type,
        )

        self._label_column = label
        self._presets = presets.lower() if presets else None
        self._validation_metric_name = validation_metric.lower() if validation_metric else None
        self._minmax_mode = None
        self._output_shape = None
        self._ckpt_path = None
        self._pretrained_path = None
        self._pretrained = pretrained
        self._config = None
        self._df_preprocessor = None
        self._column_types = None
        self._data_processors = None
        self._model_postprocess_fn = None
        self._model = None
        self._resume = False
        self._verbosity = verbosity
        self._warn_if_exist = warn_if_exist
        self._enable_progress_bar = enable_progress_bar if enable_progress_bar is not None else True
        self._fit_called = False  # Flag whether is continuous training.
        self._save_path = path
        self._hyperparameters = hyperparameters
        self._advanced_hyperparameters = None
        self._hyperparameter_tune_kwargs = None
        self._train_data = None
        self._tuning_data = None
        self._is_hpo = False
        self._teacher_learner = None
        self._fit_args = None
        # Summary statistics used in fit summary. TODO: wrap it in a class.
        self._total_train_time = None
        self._best_score = None

        self._log_filters = [
            ".*does not have many workers.* in the `DataLoader` init to improve performance.*",
            "Checkpoint directory .* exists and is not empty.",
        ]

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
    def problem_property(self):
        if self.problem_type is None:
            return None
        else:
            return PROBLEM_TYPES_REG.get(self.problem_type)

    @property
    def column_types(self):
        return self._column_types

    @property
    def eval_metric(self):
        return self._eval_metric_name

    @property
    def validation_metric(self):
        return self._validation_metric_name

    @property
    def total_parameters(self) -> int:
        return sum(p.numel() if not is_lazy_weight_tensor(p) else 0 for p in self._model.parameters())

    @property
    def trainable_parameters(self) -> int:
        return sum(
            p.numel() if not is_lazy_weight_tensor(p) else 0 for p in self._model.parameters() if p.requires_grad
        )

    @property
    def model_size(self) -> float:
        """
        Returns the model size in Megabyte.
        """
        model_size = sum(
            p.numel() * p.element_size() if not is_lazy_weight_tensor(p) else 0 for p in self._model.parameters()
        )
        return model_size * 1e-6  # convert to megabytes

    def set_eval_metric_func(self):
        from .matching import MatchingLearner
        from .ner import NERLearner
        from .object_detection import ObjectDetectionLearner
        from .semantic_segmentation import SemanticSegmentationLearner

        if (
            not isinstance(self, (NERLearner, SemanticSegmentationLearner, MatchingLearner, ObjectDetectionLearner))
            and self._eval_metric_func is None
        ):
            self._eval_metric_func = get_metric(self._eval_metric_name)

    def ensure_fit_ready(self):
        if self._problem_type and not self.problem_property.support_fit:
            raise RuntimeError(
                f"The problem_type='{self._problem_type}' does not support `learner.fit()`. "
                f"You may try to use `learner.predict()` or `learner.evaluate()`."
            )

    def infer_problem_type(self, train_data: pd.DataFrame):
        if self._fit_called:
            return
        if self._label_column:
            if isinstance(train_data, str):
                train_data = load_pd.load(train_data)
            self._problem_type = infer_problem_type(
                y_train_data=train_data[self._label_column],
                provided_problem_type=self._problem_type,
            )

    def setup_save_path(self, save_path: str):
        self._save_path = setup_save_path(
            resume=self._resume,
            old_save_path=self._save_path,
            proposed_save_path=save_path,
            raise_if_exist=True,
            warn_if_exist=False,
            fit_called=self._fit_called,
        )

    def infer_column_types(
        self, column_types: Optional[Dict] = None, data: Optional[pd.DataFrame] = None, is_train=True
    ):
        if is_train and self._fit_called:
            return
        elif is_train:
            self._column_types = infer_column_types(
                data=self._train_data,
                valid_data=self._tuning_data,
                label_columns=self._label_column,
                provided_column_types=column_types,
                problem_type=self._problem_type,  # used to update the corresponding column type
            )
            logger.debug(f"column_types: {self._column_types}")
            logger.debug(f"image columns: {[k for k, v in self._column_types.items() if v == 'image_path']}")
        elif column_types is None:
            allowable_dtypes, fallback_dtype = infer_dtypes_by_model_names(model_config=self._config.model)
            return infer_column_types(
                data=data,
                label_columns=self._label_column,
                problem_type=self._problem_type,
                allowable_column_types=allowable_dtypes,
                fallback_column_type=fallback_dtype,
            )
        else:
            return column_types

    def infer_output_shape(self):
        if self._fit_called:
            return
        self._output_shape = infer_output_shape(
            label_column=self._label_column,
            data=self._train_data,
            problem_type=self._problem_type,
        )

    def prepare_train_tuning_data(
        self,
        train_data: Union[pd.DataFrame, str],
        tuning_data: Optional[Union[pd.DataFrame, str]],
        holdout_frac: Optional[float],
        seed: Optional[int],
    ):
        if isinstance(train_data, str):
            train_data = load_pd.load(train_data)
        if isinstance(tuning_data, str):
            tuning_data = load_pd.load(tuning_data)

        if tuning_data is None:
            train_data, tuning_data = split_train_tuning_data(
                data=train_data,
                holdout_frac=holdout_frac,
                problem_type=self._problem_type,
                label_column=self._label_column,
                random_state=seed,
            )

        self._train_data = train_data
        self._tuning_data = tuning_data

    def detect_data_scarcity_mode(self):
        # Determine data scarcity mode, i.e. a few-shot scenario
        scarcity_mode = infer_scarcity_mode_by_data_size(
            df_train=self._train_data, scarcity_threshold=50
        )  # Add as separate hyperparameter somewhere?
        if scarcity_mode == FEW_SHOT and (
            not self._presets or FEW_SHOT not in self._presets
        ):  # TODO: check for data type
            logger.warning(
                f"Detected data scarcity. Consider running using the problem_type '{FEW_SHOT_CLASSIFICATION}' for better performance."
            )

    def update_attributes(
        self,
        config: Optional[Dict] = None,
        df_preprocessor: Optional[MultiModalFeaturePreprocessor] = None,
        data_processors: Optional[Dict] = None,
        model: Optional[nn.Module] = None,
        model_postprocess_fn: Optional[Callable] = None,
        best_score: Optional[float] = None,
        **kwargs,
    ):
        if config:
            self._config = config
        if df_preprocessor:
            self._df_preprocessor = df_preprocessor
        if data_processors:
            self._data_processors = data_processors
        if model:
            self._model = model
        if model_postprocess_fn:
            self._model_postprocess_fn = model_postprocess_fn
        if best_score:
            self._best_score = best_score

    def infer_validation_metric(self, is_matching: Optional[bool] = False):
        if self._fit_called:
            return
        self._validation_metric_name, self._eval_metric_name = infer_metrics(
            problem_type=self._problem_type,
            eval_metric=self._eval_metric_name if self._eval_metric_func is None else self._eval_metric_func,
            validation_metric_name=self._validation_metric_name,
            is_matching=is_matching,
        )
        self.set_eval_metric_func()
        self._minmax_mode = get_minmax_mode(self._validation_metric_name)
        logger.debug(f"validation_metric_name: {self._validation_metric_name}")
        logger.debug(f"minmax_mode: {self._minmax_mode}")

    def update_hyperparameters(self, hyperparameters: Dict, hyperparameter_tune_kwargs: Dict):
        problem_type = self._pipeline if hasattr(self, "_pipeline") else self._problem_type  # matching uses pipeline
        if self._hyperparameters and hyperparameters:
            self._hyperparameters.update(hyperparameters)
        elif hyperparameters:
            self._hyperparameters = hyperparameters

        if self._hyperparameter_tune_kwargs and hyperparameter_tune_kwargs:
            self._hyperparameter_tune_kwargs.update(hyperparameter_tune_kwargs)
        elif hyperparameter_tune_kwargs:
            self._hyperparameter_tune_kwargs = hyperparameter_tune_kwargs

        self._hyperparameters, self._hyperparameter_tune_kwargs = update_hyperparameters(
            problem_type=problem_type,
            presets=self._presets,
            provided_hyperparameters=self._hyperparameters,
            provided_hyperparameter_tune_kwargs=self._hyperparameter_tune_kwargs,
        )
        # split out the hyperparameters whose values are complex objects
        self._hyperparameters, self._advanced_hyperparameters = split_hyperparameters(self._hyperparameters)
        self._is_hpo = True if self._hyperparameter_tune_kwargs else False
        if self._is_hpo:
            try_import_ray()
            self._hyperparameters = filter_hyperparameters(
                hyperparameters=self._hyperparameters,
                column_types=self._column_types,
                config=self._config,
                fit_called=self._fit_called,
            )

    def fit_sanity_check(self):
        assert not self._resume or not self._is_hpo, "You can not resume training with HPO."
        if self._is_hpo and hasattr(self, "_teacher_learner") and self._teacher_learner is not None:
            assert isinstance(self._teacher_learner, str), (
                "HPO with distillation only supports passing a path to the learner."
            )

    def prepare_fit_args(
        self,
        time_limit: int,
        seed: int,
        standalone: Optional[bool] = True,
        clean_ckpts: Optional[bool] = True,
    ):
        if time_limit is not None:
            time_limit = timedelta(seconds=time_limit)
        self._fit_args = dict(
            max_time=time_limit,
            save_path=self._save_path,  # In HPO mode, this would be overwritten by per trial path.
            ckpt_path=None if self._is_hpo else self._ckpt_path,
            resume=False if self._is_hpo else self._resume,
            enable_progress_bar=False if self._is_hpo else self._enable_progress_bar,
            seed=seed,
            hyperparameters=self._hyperparameters,  # In HPO mode, this would be overwritten by sampled hyperparameters.
            advanced_hyperparameters=self._advanced_hyperparameters,
            standalone=standalone,
            clean_ckpts=clean_ckpts,
        )
        if self._fit_called:  # continuous training
            continuous_train_args = dict(
                config=self._config,
                df_preprocessor=self._df_preprocessor,
                data_processors=self._data_processors,
                model=self._model,
            )
            self._fit_args.update(continuous_train_args)

    def execute_fit(self):
        if self._is_hpo:
            self._fit_args["learner"] = self
            hyperparameter_tune(
                hyperparameter_tune_kwargs=self._hyperparameter_tune_kwargs,
                resources=dict(num_gpus=ResourceManager.get_gpu_count_torch()),  # TODO: allow customizing GPUs
                **self._fit_args,
            )
            return dict()
        else:
            attributes = self.fit_per_run(**self._fit_args)
            self.update_attributes(**attributes)  # only update attributes for non-HPO mode
            return attributes

    def on_fit_start(
        self,
        presets: Optional[str] = None,
        teacher_learner: Optional[Union[str, BaseLearner]] = None,
    ):
        self.ensure_fit_ready()
        if presets:
            if self._fit_called:
                warnings.warn("Ignoring the provided `presets` as fit() was called before.", UserWarning)
            else:
                self._presets = presets
        if teacher_learner:
            self._teacher_learner = teacher_learner

        logger.info(on_fit_start_message(path=self._save_path))
        training_start = time.time()
        return training_start

    def on_fit_end(
        self,
        training_start: float,
        strategy: Optional[str] = None,
        strict_loading: Optional[bool] = True,
        standalone: Optional[bool] = True,
        clean_ckpts: Optional[bool] = True,
    ):
        self._fit_called = True
        if not self._is_hpo:
            # top_k_average is called inside hyperparameter_tune() when building the final predictor.
            self.top_k_average(
                save_path=self._save_path,
                top_k_average_method=self._config.optim.top_k_average_method,
                strategy=strategy,
                strict_loading=strict_loading,
                # Not strict loading if using parameter-efficient finetuning
                standalone=standalone,
                clean_ckpts=clean_ckpts,
            )

        training_end = time.time()
        self._total_train_time = training_end - training_start
        # TODO(?) We should have a separate "_post_training_event()" for logging messages.
        logger.info(on_fit_end_message(self._save_path))

    def fit(
        self,
        train_data: Union[pd.DataFrame, str],
        presets: Optional[str] = None,
        tuning_data: Optional[Union[pd.DataFrame, str]] = None,
        time_limit: Optional[int] = None,
        save_path: Optional[str] = None,
        hyperparameters: Optional[Union[str, Dict, List[str]]] = None,
        column_types: Optional[Dict] = None,
        holdout_frac: Optional[float] = None,
        teacher_learner: Union[str, BaseLearner] = None,
        seed: Optional[int] = 0,
        standalone: Optional[bool] = True,
        hyperparameter_tune_kwargs: Optional[Dict] = None,
        clean_ckpts: Optional[bool] = True,
        **kwargs,
    ):
        self.setup_save_path(save_path=save_path)
        training_start = self.on_fit_start(presets=presets, teacher_learner=teacher_learner)
        self.infer_problem_type(train_data=train_data)
        self.prepare_train_tuning_data(
            train_data=train_data,
            tuning_data=tuning_data,
            holdout_frac=holdout_frac,
            seed=seed,
        )
        self.infer_column_types(column_types=column_types)
        self.infer_output_shape()
        self.infer_validation_metric()
        self.update_hyperparameters(
            hyperparameters=hyperparameters,
            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
        )
        self.fit_sanity_check()
        self.prepare_fit_args(
            time_limit=time_limit,
            seed=seed,
            standalone=standalone,
            clean_ckpts=clean_ckpts,
        )
        fit_returns = self.execute_fit()
        self.on_fit_end(
            training_start=training_start,
            strategy=fit_returns.get("strategy", None),
            strict_loading=fit_returns.get("strict_loading", True),
            standalone=standalone,
            clean_ckpts=clean_ckpts,
        )

        return self

    def init_pretrained(self):
        # split out the hyperparameters whose values are complex objects
        hyperparameters, advanced_hyperparameters = split_hyperparameters(self._hyperparameters)
        if self._config is None:
            self._config = get_config(
                problem_type=self._problem_type,
                presets=self._presets,
                overrides=hyperparameters,
            )
        if self._model is None:
            assert len(self._config.model.names) == 1, (
                f"Zero shot mode only supports using one model, but detects multiple models {self._config.model.names}"
            )
            self._model = create_fusion_model(
                config=self._config,
                pretrained=self._pretrained,
                num_classes=self._output_shape,
            )
        if self._data_processors is None:
            self._data_processors = create_fusion_data_processors(
                config=self._config,
                model=self._model,
                advanced_hyperparameters=advanced_hyperparameters,
            )
        self._config = self.update_strategy_by_env(config=self._config)

    def get_config_per_run(self, config, hyperparameters):
        config = get_config(
            problem_type=self._problem_type,
            presets=self._presets,
            config=config,
            overrides=hyperparameters,  # don't use self._hyperparameters due to HPO.
            extra=[DISTILLER] if self._teacher_learner is not None else None,
        )
        config = update_config_by_rules(
            problem_type=self._problem_type,
            config=config,
        )
        config = self.update_strategy_by_env(config=config)
        return config

    def get_df_preprocessor_per_run(
        self,
        df_preprocessor,
        data=None,
        config=None,
        column_types=None,
        is_train=True,
    ):
        if df_preprocessor is None:
            if is_train:
                df_preprocessor = init_df_preprocessor(
                    config=config,
                    column_types=self._column_types,
                    label_column=self._label_column,
                    train_df_x=self._train_data.drop(columns=self._label_column),
                    train_df_y=self._train_data[self._label_column],
                )
            else:
                df_preprocessor = init_df_preprocessor(
                    config=self._config,
                    column_types=column_types,
                    label_column=self._label_column,
                    train_df_x=data,
                    train_df_y=data[self._label_column] if self._label_column in data else None,
                )

        return df_preprocessor

    @staticmethod
    def update_config_by_data_per_run(config, df_preprocessor):
        # Avoid passing tabular data with many columns to MultiHeadAttention.
        # If models have additive_attention="auto", we enable it automatically for large tables.
        config = update_tabular_config_by_resources(
            config,
            num_numerical_columns=len(df_preprocessor.numerical_feature_names),
            num_categorical_columns=len(df_preprocessor.categorical_num_categories),
        )
        config = select_model(config=config, df_preprocessor=df_preprocessor)
        return config

    def get_model_per_run(self, model, config, df_preprocessor):
        if model is None:
            model = create_fusion_model(
                config=config,
                num_classes=self._output_shape,
                num_numerical_columns=len(df_preprocessor.numerical_feature_names),
                num_categories=df_preprocessor.categorical_num_categories,
                numerical_fill_values=df_preprocessor.numerical_fill_values,
            )
        return model

    @staticmethod
    def compile_model_per_run(config, model):
        if config.env.compile.turn_on:
            assert version.parse(torch.__version__) >= version.parse(TORCH_COMPILE_MIN_VERSION), (
                f"torch.compile requires torch version >= {TORCH_COMPILE_MIN_VERSION}, "
                f"but torch version {torch.__version__} is detected."
            )
            logger.debug("Using torch.compile() in compiling the model.")
            model = torch.compile(
                model,
                mode=config.env.compile.mode,
                dynamic=config.env.compile.dynamic,
                backend=config.env.compile.backend,
            )
        return model

    @staticmethod
    def get_peft_param_names_per_run(model, config):
        peft_param_names = None
        peft = config.optim.peft
        if peft:
            norm_param_names = get_norm_layer_param_names(model)
            peft_param_names = get_peft_param_names(
                norm_param_names,
                peft=peft,
            )
        return peft_param_names

    def get_data_processors_per_run(
        self,
        data_processors,
        config=None,
        model=None,
        advanced_hyperparameters=None,
        requires_label=None,
        is_train=True,
    ):
        if is_train:
            if data_processors is None:
                data_processors = create_fusion_data_processors(
                    config=config,
                    model=model,
                    advanced_hyperparameters=advanced_hyperparameters,
                )
                data_processors_count = {k: len(v) for k, v in data_processors.items()}
                logger.debug(f"data_processors_count: {data_processors_count}")
        else:
            # For each inference call, decouple the used data processors from the learner's attribute
            data_processors = copy.copy(data_processors)
            # For prediction data with no labels provided.
            if not requires_label:
                data_processors.pop(LABEL, None)

        return data_processors

    def get_validation_metric_per_run(self):
        validation_metric, custom_metric_func = get_torchmetric(
            metric_name=self._validation_metric_name,
            num_classes=self._output_shape,
            problem_type=self._problem_type,
        )
        return validation_metric, custom_metric_func

    def get_mixup_func_per_run(self, config):
        mixup_active, mixup_func = get_mixup(
            model_config=config.model,
            mixup_config=config.data.mixup,
            num_classes=self._output_shape,
        )
        if mixup_active and (config.env.per_gpu_batch_size == 1 or config.env.per_gpu_batch_size % 2 == 1):
            warnings.warn(
                "The mixup is done on the batch.The per_gpu_batch_size should be >1 and even for reasonable operation",
                UserWarning,
            )
        return mixup_active, mixup_func

    def get_loss_func_per_run(self, config, mixup_active=None):
        loss_func = get_loss_func(
            problem_type=self._problem_type,
            mixup_active=mixup_active,
            loss_func_name=config.optim.loss_func,
            config=config.optim,
        )
        aug_loss_func = get_aug_loss_func(
            config=config.optim,
            problem_type=self._problem_type,
        )
        return loss_func, aug_loss_func

    def get_model_postprocess_fn_per_run(self, loss_func):
        model_postprocess_fn = get_model_postprocess_fn(
            problem_type=self._problem_type,
            loss_func=loss_func,
        )
        return model_postprocess_fn

    def get_datamodule_per_run(
        self,
        df_preprocessor,
        data_processors,
        per_gpu_batch_size,
        num_workers,
        predict_data=None,
        is_train=True,
    ):
        if is_train and self._teacher_learner is not None:
            df_preprocessor = [df_preprocessor, self._teacher_learner._df_preprocessor]
            data_processors = [data_processors, self._teacher_learner._data_processors]
        datamodule_kwargs = dict(
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
            per_gpu_batch_size=per_gpu_batch_size,
            num_workers=num_workers,
        )
        if is_train:
            datamodule_kwargs.update(dict(train_data=self._train_data, validate_data=self._tuning_data))
        else:
            datamodule_kwargs.update(dict(predict_data=predict_data))

        datamodule = BaseDataModule(**datamodule_kwargs)
        return datamodule

    def get_optim_kwargs_per_run(
        self,
        config,
        validation_metric,
        custom_metric_func,
        loss_func,
        aug_loss_func,
        mixup_func,
        grad_steps,
    ):
        return dict(
            optim_type=config.optim.optim_type,
            lr_choice=config.optim.lr_choice,
            lr_schedule=config.optim.lr_schedule,
            lr=config.optim.lr,
            lr_decay=config.optim.lr_decay,
            end_lr=config.optim.end_lr,
            lr_mult=config.optim.lr_mult,
            weight_decay=config.optim.weight_decay,
            warmup_steps=config.optim.warmup_steps,
            track_grad_norm=config.optim.track_grad_norm,
            validation_metric=validation_metric,
            validation_metric_name=self._validation_metric_name,
            custom_metric_func=custom_metric_func,
            loss_func=loss_func,
            mixup_fn=mixup_func,
            peft=config.optim.peft,
            mixup_off_epoch=config.data.mixup.turn_off_epoch,
            skip_final_val=config.optim.skip_final_val,
            cross_modal_align=config.optim.cross_modal_align,
            cross_modal_align_weight=config.optim.cross_modal_align_weight,
            automatic_optimization=config.optim.automatic_optimization,
            accumulate_grad_batches=grad_steps,
            gradient_clip_val=config.optim.gradient_clip_val,
            gradient_clip_algorithm=config.optim.gradient_clip_algorithm,
            use_aug_optim=config.optim.lemda.turn_on,
            aug_loss_func=aug_loss_func,
            aug_lr=config.optim.lemda.lr,
            aug_weight_decay=config.optim.lemda.weight_decay,
            aug_optim_type=config.optim.lemda.optim_type,
        )

    def get_litmodule_per_run(
        self,
        model=None,
        model_postprocess_fn=None,
        peft_param_names=None,
        optim_kwargs=dict(),
        distillation_kwargs=dict(),
        is_train=True,
    ):
        if is_train:
            if self._teacher_learner is not None:
                return DistillerLitModule(
                    student_model=model,
                    teacher_model=self._teacher_learner._model,
                    **optim_kwargs,
                    **distillation_kwargs,
                )
            else:
                return LitModule(
                    model=model,
                    model_postprocess_fn=model_postprocess_fn,
                    trainable_param_names=peft_param_names,
                    **optim_kwargs,
                )
        else:
            return LitModule(
                model=self._model,
                model_postprocess_fn=self._model_postprocess_fn,
                **optim_kwargs,
            )

    def get_callbacks_per_run(self, save_path=None, config=None, litmodule=None, pred_writer=None, is_train=True):
        if not is_train:
            if pred_writer is not None:
                callbacks = [pred_writer]
            else:
                callbacks = []
            return callbacks

        checkpoint_callback = AutoMMModelCheckpoint(
            dirpath=save_path,
            save_top_k=config.optim.top_k,
            verbose=True,
            monitor=litmodule.validation_metric_name,
            mode=self._minmax_mode,
            save_last=True,
        )
        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor=litmodule.validation_metric_name,
            patience=config.optim.patience,
            mode=self._minmax_mode,
            stopping_threshold=get_stopping_threshold(self._validation_metric_name),
        )
        lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
        model_summary = pl.callbacks.ModelSummary(max_depth=1)
        callbacks = [
            checkpoint_callback,
            early_stopping_callback,
            lr_callback,
            model_summary,
        ]
        if self._is_hpo:
            from ..utils.hpo import get_ray_tune_ckpt_callback

            TuneReportCheckpointCallback = get_ray_tune_ckpt_callback()
            tune_report_callback = TuneReportCheckpointCallback(
                {f"{litmodule.validation_metric_name}": f"{litmodule.validation_metric_name}"},
                filename=RAY_TUNE_CHECKPOINT,
            )
            callbacks = [
                tune_report_callback,
                early_stopping_callback,
                lr_callback,
                model_summary,
            ]

        return callbacks

    @staticmethod
    def get_plugins_per_run(model, peft_param_names=None):
        custom_checkpoint_plugin = AutoMMModelCheckpointIO(
            trainable_param_names=peft_param_names,
            model_name_to_id=model.name_to_id,
        )
        return [custom_checkpoint_plugin]

    @staticmethod
    def get_tb_logger(save_path):
        return pl.loggers.TensorBoardLogger(
            save_dir=save_path,
            name="",
            version="",
        )

    @staticmethod
    def log_gpu_info(num_gpus, config):
        logger.info(
            get_gpu_message(
                detected_num_gpus=ResourceManager.get_gpu_count_torch(),
                used_num_gpus=num_gpus,
                strategy=config.env.strategy,
            )
        )

    @staticmethod
    def get_grad_steps(num_gpus, config):
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
        return grad_steps

    @staticmethod
    def get_strategy_per_run(num_gpus, config):
        if (
            config.env.strategy == DEEPSPEED_OFFLOADING and num_gpus == 1 and DEEPSPEED_MODULE not in sys.modules
        ):  # Offloading currently only tested for single GPU
            assert version.parse(pl.__version__) >= version.parse(DEEPSPEED_MIN_PL_VERSION), (
                f"For DeepSpeed Offloading to work reliably you need at least lightning version {DEEPSPEED_MIN_PL_VERSION}, however, found {pl.__version__}. Please update your lightning version."
            )
            from ..optim.deepspeed import CustomDeepSpeedStrategy

            strategy = CustomDeepSpeedStrategy(
                stage=3,
                offload_optimizer=True,
                offload_parameters=False,
                allgather_bucket_size=config.env.deepspeed_allgather_size,
                reduce_bucket_size=config.env.deepspeed_allreduce_size,
            )
        elif num_gpus <= 1:
            strategy = "auto"
        else:
            strategy = config.env.strategy
        return strategy

    def update_strategy_and_num_gpus_for_hpo(self, strategy, num_gpus):
        if self._is_hpo:
            strategy = "auto"
            num_gpus = min(num_gpus, 1)  # Currently only support one trial using one gpu.
        return strategy, num_gpus

    @staticmethod
    def get_precision_per_run(num_gpus: int, precision: Union[str, int], cpu_only_warning: Optional[bool] = True):
        return infer_precision(num_gpus=num_gpus, precision=precision, cpu_only_warning=cpu_only_warning)

    def get_num_gpus_and_strategy_per_run(
        self,
        config: Optional[DictConfig] = None,
        predict_data: Optional[pd.DataFrame] = None,
        is_train: Optional[bool] = True,
    ):
        if is_train:
            data = self._train_data
        else:
            data = predict_data
            config = self._config

        num_gpus = compute_num_gpus(
            config_num_gpus=config.env.num_gpus,
            accelerator=config.env.accelerator,
        )
        num_gpus = self.update_num_gpus_by_data_size(num_gpus=num_gpus, data=data)
        strategy = self.get_strategy_per_run(num_gpus=num_gpus, config=config)
        strategy, num_gpus = self.update_strategy_and_num_gpus_for_hpo(strategy=strategy, num_gpus=num_gpus)
        num_gpus, strategy = run_ddp_only_once(num_gpus=num_gpus, strategy=strategy)

        if is_train:
            self.log_gpu_info(num_gpus=num_gpus, config=config)

        return num_gpus, strategy

    @staticmethod
    def post_update_config_per_run(config, num_gpus, precision, strategy):
        config.env.num_gpus = num_gpus
        config.env.precision = precision
        # for deepspeed offloading, the strategy becomes is a customized strategy object instead of a string,
        # but config still needs a string.
        config.env.strategy = strategy if not config.env.strategy == DEEPSPEED_OFFLOADING else DEEPSPEED_OFFLOADING
        return config

    def init_trainer_per_run(
        self,
        num_gpus,
        precision,
        strategy,
        callbacks,
        max_time=None,
        config=None,
        tb_logger=None,
        grad_steps=None,
        plugins=None,
        enable_progress_bar=None,
        barebones=False,
        is_train=True,
    ):
        if is_train:
            trainer_kwargs = dict(
                accelerator="gpu" if num_gpus > 0 else config.env.accelerator,
                devices=num_gpus if num_gpus > 0 else "auto",
                num_nodes=config.env.num_nodes,
                precision=precision,
                strategy=strategy if strategy else "auto",
                benchmark=False,
                deterministic=config.env.deterministic,
                max_epochs=config.optim.max_epochs,
                max_steps=config.optim.max_steps,
                max_time=max_time,
                callbacks=callbacks,
                logger=tb_logger,
                log_every_n_steps=config.optim.log_every_n_steps,
                enable_progress_bar=enable_progress_bar,
                fast_dev_run=config.env.fast_dev_run,
                val_check_interval=config.optim.val_check_interval,
                check_val_every_n_epoch=config.optim.check_val_every_n_epoch,
                plugins=plugins,
            )
            if config.optim.automatic_optimization:
                trainer_kwargs.update(
                    dict(
                        gradient_clip_val=config.optim.gradient_clip_val,
                        gradient_clip_algorithm=config.optim.gradient_clip_algorithm,
                        accumulate_grad_batches=grad_steps,
                    )
                )
            blacklist_msgs = ["already configured with model summary"]
            log_filter = LogFilter(blacklist_msgs)
            with apply_log_filter(log_filter):
                trainer = pl.Trainer(**trainer_kwargs)
        else:
            blacklist_msgs = []
            if self._verbosity <= 3:  # turn off logging in prediction
                blacklist_msgs.append("Automatic Mixed Precision")
                blacklist_msgs.append("GPU available")
                blacklist_msgs.append("TPU available")
                blacklist_msgs.append("IPU available")
                blacklist_msgs.append("HPU available")
                blacklist_msgs.append("select gpus")
                blacklist_msgs.append("Trainer(barebones=True)")
            log_filter = LogFilter(blacklist_msgs)

            with apply_log_filter(log_filter):
                trainer = pl.Trainer(
                    accelerator="gpu" if num_gpus > 0 else self._config.env.accelerator,
                    devices=num_gpus if num_gpus > 0 else "auto",
                    num_nodes=self._config.env.num_nodes,
                    precision=precision,
                    strategy=strategy,
                    benchmark=False,
                    enable_progress_bar=False if barebones else self._enable_progress_bar,
                    deterministic=self._config.env.deterministic,
                    max_epochs=-1,  # Add max_epochs to disable warning
                    logger=False,
                    callbacks=callbacks,
                    barebones=barebones,
                )

        return trainer

    def run_trainer(
        self,
        trainer,
        litmodule,
        datamodule,
        ckpt_path=None,
        resume=None,
        pred_writer=None,
        is_train=True,
    ):
        with warnings.catch_warnings():
            for filter in self._log_filters:
                warnings.filterwarnings("ignore", filter)
            if is_train:
                trainer.fit(
                    litmodule,
                    datamodule=datamodule,
                    ckpt_path=ckpt_path if resume else None,  # this is to resume training that was broken accidentally
                )
            else:
                blacklist_msgs = []
                if self._verbosity <= 3:  # turn off logging in prediction
                    blacklist_msgs.append("LOCAL_RANK")
                    blacklist_msgs.append("Trainer(barebones=True)")
                log_filter = LogFilter(blacklist_msgs)
                with apply_log_filter(log_filter):
                    outputs = trainer.predict(
                        litmodule,
                        datamodule=datamodule,
                        return_predictions=pred_writer is None,
                    )
                return outputs

    def on_fit_per_run_start(self, seed, save_path):
        # TODO(?) We should have a separate "_pre_training_event()" for logging messages.
        logger.info(on_fit_per_run_start_message(save_path, self._validation_metric_name))
        pl.seed_everything(seed, workers=True)

    def on_fit_per_run_end(
        self,
        trainer: pl.Trainer,
        config: DictConfig,
        model: nn.Module,
        df_preprocessor: MultiModalFeaturePreprocessor,
        data_processors: Dict,
        save_path: str,
        standalone: bool,
    ):
        self.clean_trainer_processes(trainer=trainer, is_train=True)
        self.save(
            path=save_path,
            standalone=standalone,
            config=config,
            model=model,
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
            fit_called=True,  # fit is called on one run.
            save_model=False,  # The final model will be saved in top_k_average
        )

    def fit_per_run(
        self,
        max_time: timedelta,
        save_path: str,
        ckpt_path: str,
        resume: bool,
        enable_progress_bar: bool,
        seed: int,
        hyperparameters: Optional[Union[str, Dict, List[str]]] = None,
        advanced_hyperparameters: Optional[Dict] = None,
        config: Optional[Dict] = None,
        df_preprocessor: Optional[MultiModalFeaturePreprocessor] = None,
        data_processors: Optional[Dict] = None,
        model: Optional[nn.Module] = None,
        standalone: bool = True,
        clean_ckpts: bool = True,
    ):
        self.on_fit_per_run_start(seed=seed, save_path=save_path)
        config = self.get_config_per_run(config=config, hyperparameters=hyperparameters)
        df_preprocessor = self.get_df_preprocessor_per_run(
            df_preprocessor=df_preprocessor,
            config=config,
        )
        config = self.update_config_by_data_per_run(config=config, df_preprocessor=df_preprocessor)
        model = self.get_model_per_run(model=model, config=config, df_preprocessor=df_preprocessor)
        model = self.compile_model_per_run(config=config, model=model)
        peft_param_names = self.get_peft_param_names_per_run(model=model, config=config)
        data_processors = self.get_data_processors_per_run(
            data_processors=data_processors,
            config=config,
            model=model,
            advanced_hyperparameters=advanced_hyperparameters,
        )
        validation_metric, custom_metric_func = self.get_validation_metric_per_run()
        mixup_active, mixup_func = self.get_mixup_func_per_run(config=config)
        loss_func, aug_loss_func = self.get_loss_func_per_run(config=config, mixup_active=mixup_active)
        model_postprocess_fn = self.get_model_postprocess_fn_per_run(loss_func=loss_func)
        num_gpus, strategy = self.get_num_gpus_and_strategy_per_run(config=config)
        precision = self.get_precision_per_run(num_gpus=num_gpus, precision=config.env.precision)
        grad_steps = self.get_grad_steps(num_gpus=num_gpus, config=config)

        if max_time == timedelta(seconds=0):
            return dict(
                config=config,
                df_preprocessor=df_preprocessor,
                data_processors=data_processors,
                model=model,
                model_postprocess_fn=model_postprocess_fn,
                strict_loading=not peft_param_names,
            )
        # setup distillation in each fit_per_run call to support distillation + HPO
        distillation_kwargs = self.setup_distillation(
            model=model,
            loss_func=loss_func,
            config=config,
            data_processors=data_processors,
        )
        datamodule = self.get_datamodule_per_run(
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
            per_gpu_batch_size=config.env.per_gpu_batch_size,
            num_workers=config.env.num_workers,
        )
        optim_kwargs = self.get_optim_kwargs_per_run(
            config=config,
            validation_metric=validation_metric,
            custom_metric_func=custom_metric_func,
            loss_func=loss_func,
            aug_loss_func=aug_loss_func,
            mixup_func=mixup_func,
            grad_steps=grad_steps,
        )
        litmodule = self.get_litmodule_per_run(
            model=model,
            model_postprocess_fn=model_postprocess_fn,
            peft_param_names=peft_param_names,
            optim_kwargs=optim_kwargs,
            distillation_kwargs=distillation_kwargs,
        )
        callbacks = self.get_callbacks_per_run(save_path=save_path, config=config, litmodule=litmodule)
        plugins = self.get_plugins_per_run(model=model, peft_param_names=peft_param_names)
        tb_logger = self.get_tb_logger(save_path=save_path)
        config = self.post_update_config_per_run(
            config=config,
            num_gpus=num_gpus,
            precision=precision,
            strategy=strategy,
        )
        trainer = self.init_trainer_per_run(
            num_gpus=num_gpus,
            config=config,
            precision=precision,
            strategy=strategy,
            max_time=max_time,
            callbacks=callbacks,
            tb_logger=tb_logger,
            grad_steps=grad_steps,
            plugins=plugins,
            enable_progress_bar=enable_progress_bar,
        )

        self.run_trainer(
            trainer=trainer,
            litmodule=litmodule,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
            resume=resume,
        )
        self.on_fit_per_run_end(
            save_path=save_path,
            standalone=standalone,
            trainer=trainer,
            config=config,
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
            model=model,
        )

        best_score = (
            trainer.callback_metrics[f"val_{self._validation_metric_name}"].item()
            if f"val_{self._validation_metric_name}" in trainer.callback_metrics
            else self._best_score
        )  # https://github.com/autogluon/autogluon/issues/4428

        return dict(
            config=config,
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
            model=model,
            model_postprocess_fn=model_postprocess_fn,
            best_score=best_score,
            strategy=strategy,
            strict_loading=not peft_param_names,
        )

    def top_k_average(
        self,
        save_path,
        top_k_average_method,
        strategy=None,
        last_ckpt_path=None,
        strict_loading=True,
        standalone=True,
        clean_ckpts=True,
    ):
        eval_metric = self._eval_metric_name if self._eval_metric_func is None else self._eval_metric_func
        minmax_mode = get_minmax_mode(eval_metric)
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

        if self._teacher_learner is not None:
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

                        self._load_state_dict(
                            path=top_k_model_paths[0],
                            prefix=prefix,
                            strict=strict_loading,
                        )
                        best_score = self.evaluate(self._tuning_data, metrics=[eval_metric])
                        best_score = next(iter(best_score.values()))
                        for i in range(1, len(top_k_model_paths)):
                            cand_avg_state_dict = average_checkpoints(
                                checkpoint_paths=ingredients + [top_k_model_paths[i]],
                            )
                            self._load_state_dict(
                                state_dict=cand_avg_state_dict,
                                prefix=prefix,
                                strict=strict_loading,
                            )
                            cand_score = self.evaluate(self._tuning_data, metrics=[eval_metric])
                            cand_score = next(iter(cand_score.values()))
                            if monitor_op(cand_score, best_score):
                                # Add new ingredient
                                ingredients.append(top_k_model_paths[i])
                                best_score = cand_score
                elif top_k_average_method == BEST:
                    ingredients = [top_k_model_paths[0]]
                else:
                    raise ValueError(
                        f"The key for 'optim.top_k_average_method' is not supported. "
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
        self._load_state_dict(
            state_dict=avg_state_dict,
            prefix=prefix,
            strict=strict_loading,
        )

        if self._teacher_learner is not None:
            avg_state_dict = self._replace_model_name_prefix(
                state_dict=avg_state_dict,
                old_prefix="student_model",
                new_prefix="model",
            )

        if not standalone:
            checkpoint = {"state_dict": avg_state_dict}
        else:
            if isinstance(strategy, DeepSpeedStrategy):
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

        torch.save(checkpoint, os.path.join(save_path, MODEL_CHECKPOINT))  # nosec B614

        if clean_ckpts:
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

    def prepare_deepspeed_offloading(self, strategy):
        # TODO: Using optimiation_kwargs for inference is confusing and bad design. Remove as soon as fixed in lightning.
        if self._config.env.strategy == DEEPSPEED_OFFLOADING and DEEPSPEED_MODULE not in sys.modules:
            # Need to initialize DeepSpeed and optimizer as currently required in lightning's integration of deepspeed.
            from ..optim.deepspeed import CustomDeepSpeedStrategy

            strategy = CustomDeepSpeedStrategy(
                stage=3,
                offload_optimizer=True,
                offload_parameters=False,
                allgather_bucket_size=self._config.env.deepspeed_allgather_size,
                reduce_bucket_size=self._config.env.deepspeed_allreduce_size,
            )

            optim_kwargs = dict(
                optim_type=self._config.optim.optim_type,
                lr_choice=self._config.optim.lr_choice,
                lr_schedule=self._config.optim.lr_schedule,
                lr=self._config.optim.lr,
                lr_decay=self._config.optim.lr_decay,
                end_lr=self._config.optim.end_lr,
                lr_mult=self._config.optim.lr_mult,
                weight_decay=self._config.optim.weight_decay,
                warmup_steps=self._config.optim.warmup_steps,
            )
        else:
            optim_kwargs = {}

        return strategy, optim_kwargs

    def get_pred_writer(self, strategy):
        pred_writer = None
        if isinstance(strategy, str) and DDP in strategy:
            pred_writer = DDPPredictionWriter(output_dir=self._save_path, write_interval="epoch", strategy=strategy)
        return pred_writer

    @staticmethod
    def collect_predictions(outputs, trainer, pred_writer, num_gpus):
        if pred_writer is not None:
            if trainer.global_rank == 0:
                outputs = pred_writer.collect_all_gpu_results(num_gpus=num_gpus)

        return outputs

    @staticmethod
    def clean_trainer_processes(trainer, is_train=True):
        if is_train:
            msg = "Training finished,"
        else:
            msg = "Prediction finished,"
        if trainer.global_rank != 0:
            sys.exit(f"{msg} exit the process with global_rank={trainer.global_rank}...")

    def update_image_column_types(self, data):
        column_types = self._column_types
        column_types_copy = copy.deepcopy(column_types)
        for col_name, col_type in column_types.items():
            if col_type in [IMAGE_BYTEARRAY, IMAGE_PATH, IMAGE_BASE64_STR]:
                if is_image_column(data=data[col_name], col_name=col_name, image_type=IMAGE_PATH):
                    image_type = IMAGE_PATH
                elif is_image_column(
                    data=data[col_name],
                    col_name=col_name,
                    image_type=IMAGE_BYTEARRAY,
                ):
                    image_type = IMAGE_BYTEARRAY
                elif is_image_column(data=data[col_name], col_name=col_name, image_type=IMAGE_BASE64_STR):
                    image_type = IMAGE_BASE64_STR
                else:
                    image_type = col_type
                if col_type != image_type:
                    column_types_copy[col_name] = image_type
        return column_types_copy

    def data_to_df(self, data):
        if self._fit_called:
            column_names = list(self._column_types.keys())
            # remove label column since it's not required in inference.
            column_names.remove(self._label_column)
            data = data_to_df(
                data=data,
                required_columns=self._df_preprocessor.required_feature_names,
                all_columns=column_names,
            )
        else:
            data = data_to_df(data=data)

        return data

    @staticmethod
    def update_realtime_for_interactive_env(realtime: bool, num_gpus: int, barebones: bool, strategy: str):
        # TODO: support realtime inference for notebook with multi-gpus
        # realtime can initialize CUDA, which can cause failures when calling fit again in the interactive env.
        if is_interactive_strategy(strategy) and realtime:
            realtime = False
            num_gpus = min(1, num_gpus)
            barebones = True

        return realtime, num_gpus, barebones

    @staticmethod
    def update_num_gpus_by_data_size(
        num_gpus: int,
        data: pd.DataFrame,
    ):
        data_size = len(data)
        if data_size < num_gpus:
            num_gpus = data_size
        return num_gpus

    def realtime_predict(
        self,
        data: pd.DataFrame,
        df_preprocessor: Union[MultiModalFeaturePreprocessor, List[MultiModalFeaturePreprocessor]],
        data_processors: Union[Dict, List[Dict]],
        num_gpus: int,
        precision: Union[int, str],
    ) -> List[Dict]:
        """
        Perform realtime inference.

        Parameters
        ----------
        data
            A dataframe.
        df_preprocessor
            Dataframe preprocessors.
        data_processors
            Data processors.
        num_gpus
            Number of GPUs.
        precision
            The precision used in inference.

        Returns
        -------
        A list of output dicts.
        """

        batch = self.process_batch(
            data=data,
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
        )
        output = self.predict_batch(
            batch=batch,
            model=self._model,
            precision=precision,
            num_gpus=num_gpus,
        )
        return [output]

    def on_predict_per_run_start(self, data: Union[str, pd.DataFrame]):
        data = self.data_to_df(data=data)
        return data

    def get_predict_batch_size_per_run(self, num_gpus: int, strategy: str):
        return compute_inference_batch_size(
            per_gpu_batch_size=self._config.env.per_gpu_batch_size,
            inference_batch_size_ratio=self._config.env.inference_batch_size_ratio,
            num_gpus=num_gpus,
            strategy=strategy,
        )

    def on_predict_per_run_end(self, trainer):
        self.clean_trainer_processes(trainer=trainer, is_train=False)

    def predict_per_run(
        self,
        data: Union[pd.DataFrame, dict, list],
        realtime: Optional[bool],
        requires_label: Optional[bool] = False,
        barebones: Optional[bool] = False,
    ) -> List[Dict]:
        """
        Perform inference for learner.

        Parameters
        ----------
        data
            The data for inference.
        realtime
            Whether use realtime inference.
        requires_label
            Whether uses label during inference.
        barebones
            Whether to run in “barebones mode”, where all lightning's features that may impact raw speed are disabled.

        Returns
        -------
        A list of output dicts.
        """
        data = self.on_predict_per_run_start(data=data)
        column_types = self.infer_column_types(
            column_types=self._column_types,
            data=data,
            is_train=False,
        )
        df_preprocessor = self.get_df_preprocessor_per_run(
            df_preprocessor=self._df_preprocessor,
            data=data,
            column_types=column_types,
            is_train=False,
        )
        if self._fit_called:
            df_preprocessor._column_types = self.update_image_column_types(data=data)
        data_processors = self.get_data_processors_per_run(
            data_processors=self._data_processors,
            requires_label=requires_label,
            is_train=False,
        )
        num_gpus, strategy = self.get_num_gpus_and_strategy_per_run(
            predict_data=data,
            is_train=False,
        )
        precision = self.get_precision_per_run(
            num_gpus=num_gpus,
            precision=self._config.env.precision,
            cpu_only_warning=False,
        )
        batch_size = self.get_predict_batch_size_per_run(num_gpus=num_gpus, strategy=strategy)
        realtime = self.use_realtime(
            realtime=realtime,
            data=data,
            data_processors=data_processors,
            batch_size=batch_size,
        )
        realtime, num_gpus, barebones = self.update_realtime_for_interactive_env(
            realtime=realtime,
            num_gpus=num_gpus,
            barebones=barebones,
            strategy=strategy,
        )

        if realtime:
            outputs = self.realtime_predict(
                data=data,
                df_preprocessor=df_preprocessor,
                data_processors=data_processors,
                num_gpus=num_gpus,
                precision=precision,
            )
            return outputs

        strategy, optim_kwargs = self.prepare_deepspeed_offloading(strategy=strategy)
        datamodule = self.get_datamodule_per_run(
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
            per_gpu_batch_size=batch_size,
            num_workers=self._config.env.num_workers_inference,
            predict_data=data,
            is_train=False,
        )
        pred_writer = self.get_pred_writer(strategy=strategy)
        callbacks = self.get_callbacks_per_run(pred_writer=pred_writer, is_train=False)
        # TODO: remove optim_kwargs from inference
        litmodule = self.get_litmodule_per_run(optim_kwargs=optim_kwargs, is_train=False)
        trainer = self.init_trainer_per_run(
            num_gpus=num_gpus,
            precision=precision,
            strategy=strategy,
            callbacks=callbacks,
            barebones=barebones,
            is_train=False,
        )
        outputs = self.run_trainer(
            trainer=trainer,
            litmodule=litmodule,
            datamodule=datamodule,
            pred_writer=pred_writer,
            is_train=False,
        )
        outputs = self.collect_predictions(
            outputs=outputs,
            trainer=trainer,
            pred_writer=pred_writer,
            num_gpus=num_gpus,
        )
        self.on_predict_per_run_end(trainer=trainer)

        return outputs

    def ensure_predict_ready(self):
        if not self._fit_called:
            if not self._problem_type or not self.problem_property.support_zero_shot:
                raise RuntimeError(
                    f"problem_type='{self._problem_type}' does not support running inference directly. "
                    f"You need to call `learner.fit()`, or load a learner first before "
                    f"running `learner.predict()`, `learner.evaluate()` or `learner.extract_embedding()`."
                )
            else:
                self.init_pretrained()

    def on_predict_start(self):
        self.ensure_predict_ready()

    def evaluate(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        metrics: Optional[Union[str, List[str]]] = None,
        return_pred: Optional[bool] = False,
        realtime: Optional[bool] = False,
        **kwargs,
    ):
        """
        Evaluate model on a test dataset.

        Parameters
        ----------
        data
            A dataframe, containing the same columns as the training data.
        metrics
            A list of metric names to report.
            If None, we only return the score for the stored `_eval_metric_name`.
        return_pred
            Whether to return the prediction result of each row.
        realtime
            Whether to do realtime inference, which is efficient for small data (default False).
            If provided None, we would infer it on based on the data modalities
            and sample number.

        Returns
        -------
        A dictionary with the metric names and their corresponding scores.
        Optionally return a dataframe of prediction results.
        """
        self.on_predict_start()
        ret_type = LOGITS
        outputs = self.predict_per_run(
            data=data,
            realtime=realtime,
            requires_label=True,
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
        y_true = self._df_preprocessor.transform_label_for_metric(df=data)
        metric_data.update(
            {
                Y_PRED: y_pred,
                Y_TRUE: y_true,
            }
        )
        if metrics is None:
            if self._eval_metric_func:
                metrics = [self._eval_metric_func]
            else:
                metrics = [self._eval_metric_name]
        if isinstance(metrics, str) or isinstance(metrics, Scorer):
            metrics = [metrics]

        results = {}
        for per_metric in metrics:
            score = compute_score(
                metric_data=metric_data,
                metric=per_metric.lower() if isinstance(per_metric, str) else per_metric,
            )
            per_metric_name = per_metric if isinstance(per_metric, str) else per_metric.name
            results[per_metric_name] = score

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
        assert len(query_embeddings) == 1, (
            f"Multiple embedding types `{query_embeddings.keys()}` exist in query data. Please reduce them to one type."
        )
        query_embeddings = list(query_embeddings.values())[0]

        candidate_embeddings = self.extract_embedding(candidate_data, as_tensor=True)
        assert len(candidate_embeddings) == 1, (
            f"Multiple embedding types `{candidate_embeddings.keys()}` exist in candidate data. Please reduce them to one type."
        )
        candidate_embeddings = list(candidate_embeddings.values())[0]

        if return_prob:
            ret = (100.0 * query_embeddings @ candidate_embeddings.T).float().softmax(dim=-1)
        else:
            ret = (query_embeddings @ candidate_embeddings.T).argmax(dim=-1)

        ret = tensor_to_ndarray(ret)

        return ret

    def predict(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        candidate_data: Optional[Union[pd.DataFrame, dict, list]] = None,
        as_pandas: Optional[bool] = None,
        realtime: Optional[bool] = False,
        **kwargs,
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
            Whether to do realtime inference, which is efficient for small data (default False).
            If provided None, we would infer it on based on the data modalities
            and sample number.

        Returns
        -------
        Array of predictions, one corresponding to each row in given dataset.
        """
        self.on_predict_start()
        ret_type = LOGITS
        if candidate_data:
            pred = self._match_queries_and_candidates(
                query_data=data,
                candidate_data=candidate_data,
                return_prob=False,
            )
        else:
            outputs = self.predict_per_run(
                data=data,
                realtime=realtime,
                requires_label=False,
            )
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
        realtime: Optional[bool] = False,
        **kwargs,
    ):
        """
        Predict probabilities class probabilities rather than class labels.
        This is only for the classification. Calling it for regression will throw an exception.

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
            Whether to do realtime inference, which is efficient for small data (default False).
            If provided None, we would infer it on based on the data modalities
            and sample number.

        Returns
        -------
        Array of predicted class-probabilities, corresponding to each row in the given data.
        When as_multiclass is True, the output will always have shape (#samples, #classes).
        Otherwise, the output will have shape (#samples,)
        """
        self.on_predict_start()
        assert self._problem_type not in [
            REGRESSION,
        ], f"Problem {self._problem_type} has no probability output."

        if candidate_data:
            prob = self._match_queries_and_candidates(
                query_data=data,
                candidate_data=candidate_data,
                return_prob=True,
            )
        else:
            outputs = self.predict_per_run(
                data=data,
                realtime=realtime,
                requires_label=False,
            )
            logits = extract_from_output(outputs=outputs, ret_type=LOGITS)
            prob = logits_to_prob(logits)

        if not as_multiclass:
            if self._problem_type == BINARY:
                prob = prob[:, 1]

        if (as_pandas is None and isinstance(data, pd.DataFrame)) or as_pandas is True:
            prob = self._as_pandas(data=data, to_be_converted=prob)

        return prob

    def extract_embedding(
        self,
        data: Union[pd.DataFrame, dict, list],
        return_masks: Optional[bool] = False,
        as_tensor: Optional[bool] = False,
        as_pandas: Optional[bool] = False,
        realtime: Optional[bool] = False,
        **kwargs,
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
            Whether to do realtime inference, which is efficient for small data (default False).
            If provided None, we would infer it on based on the data modalities
            and sample number.

        Returns
        -------
        Array of embeddings, corresponding to each row in the given data.
        It will have shape (#samples, D) where the embedding dimension D is determined
        by the neural network's architecture.
        """
        self.on_predict_start()
        turn_on_off_feature_column_info(
            data_processors=self._data_processors,
            flag=True,
        )
        outputs = self.predict_per_run(
            data=data,
            realtime=realtime,
            requires_label=False,
        )
        if self._problem_type in [FEATURE_EXTRACTION, ZERO_SHOT_IMAGE_CLASSIFICATION]:
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

    def _load_state_dict(
        self,
        state_dict: dict = None,
        path: str = None,
        prefix: str = "model.",
        strict: bool = True,
    ):
        if state_dict is None:
            if os.path.isdir(path + "-dir"):  # deepspeed save checkpoints into a directory
                from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

                convert_zero_checkpoint_to_fp32_state_dict(path + "-dir", path)
                shutil.rmtree(path + "-dir")
                state_dict = torch.load(path, map_location=torch.device("cpu"), weights_only=False)["state_dict"]  # nosec B614
            else:
                state_dict = torch.load(path, map_location=torch.device("cpu"), weights_only=False)["state_dict"]  # nosec B614
        state_dict = {k.partition(prefix)[2]: v for k, v in state_dict.items() if k.startswith(prefix)}

        # Some buffers like `position_ids` are registered as persistent=False since transformers 4.31.0
        # Refer to https://github.com/huggingface/transformers/pull/24505/files
        buffer_names = [k for k, v in self._model.named_buffers()]
        buffer_names_to_filter = [k for k in buffer_names if k not in self._model.state_dict().keys()]
        state_dict = {k: v for k, v in state_dict.items() if k not in buffer_names_to_filter}

        load_result = self._model.load_state_dict(state_dict, strict=strict)
        assert len(load_result.unexpected_keys) == 0, (
            f"Load model failed, unexpected keys {load_result.unexpected_keys.__str__()}"
        )

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

    def save(
        self,
        path: str,
        standalone: Optional[bool] = True,
        config: Optional[DictConfig] = None,
        model: Optional[nn.Module] = None,
        df_preprocessor: Optional[MultiModalFeaturePreprocessor] = None,
        data_processors: Optional[Dict] = None,
        fit_called: Optional[bool] = None,
        save_model: Optional[bool] = True,
    ):
        """
        Save this learner to file in directory specified by `path`.

        Parameters
        ----------
        path
            The directory to save this learner.
        standalone
            Whether to save the downloaded model for offline deployment.
            When standalone = True, save the transformers.CLIPModel and transformers.AutoModel to os.path.join(path,model_name),
            and reset the associate model.model_name.checkpoint_name start with `local://` in config.yaml.
            When standalone = False, the saved artifact may require an online environment to process in load().
        """
        config = config if config else self._config
        config = copy.deepcopy(config)
        model = model if model else self._model
        if standalone and not config.optim.peft:
            config = save_pretrained_model_configs(model=model, config=config, path=path)
        os.makedirs(path, exist_ok=True)
        OmegaConf.save(config=config, f=os.path.join(path, "config.yaml"))

        df_preprocessor = df_preprocessor if df_preprocessor else self._df_preprocessor
        with open(os.path.join(path, "df_preprocessor.pkl"), "wb") as fp:
            pickle.dump(df_preprocessor, fp)

        data_processors = data_processors if data_processors else self._data_processors
        data_processors = copy.deepcopy(data_processors)

        # Save text tokenizers before saving data processors
        for modality in [TEXT, TEXT_NER, NER, DOCUMENT]:
            if modality in data_processors:
                for per_processor in data_processors[modality]:
                    per_processor.save_tokenizer(path)

        # Clear the documents cache dictionary before saving.
        for modality in [DOCUMENT]:
            if modality in data_processors:
                for p in data_processors[modality]:
                    p.documents.clear()

        with open(os.path.join(path, "data_processors.pkl"), "wb") as fp:
            pickle.dump(data_processors, fp)

        with open(os.path.join(path, "eval_metric.pkl"), "wb") as fp:
            pickle.dump(self._eval_metric_func, fp)

        with open(os.path.join(path, f"assets.json"), "w") as fp:
            json.dump(
                {
                    "learner_class": self.__class__.__name__,
                    "column_types": self._column_types,
                    "label_column": self._label_column,
                    "problem_type": self._problem_type,
                    "presets": self._presets,
                    "eval_metric_name": self._eval_metric_name,
                    "validation_metric_name": self._validation_metric_name,
                    "minmax_mode": self._minmax_mode,
                    "output_shape": self._output_shape,
                    "save_path": path,
                    "pretrained": self._pretrained,
                    "pretrained_path": self._pretrained_path,
                    "fit_called": fit_called if fit_called is not None else self._fit_called,
                    "best_score": self._best_score,
                    "total_train_time": self._total_train_time,
                    "version": ag_version.__version__,
                },
                fp,
                ensure_ascii=True,
            )

        if save_model:
            checkpoint = {"state_dict": {"model." + name: param for name, param in model.state_dict().items()}}
            torch.save(checkpoint, os.path.join(os.path.abspath(path), MODEL_CHECKPOINT))  # nosec B614

    @staticmethod
    def _load_metadata(
        learner: BaseLearner,
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
            df_preprocessor = pickle.load(fp)  # nosec B301
        try:
            with open(os.path.join(path, "data_processors.pkl"), "rb") as fp:
                data_processors = pickle.load(fp)  # nosec B301
            # Load text tokenizers after loading data processors.
            for modality in [TEXT, TEXT_NER, NER, DOCUMENT]:
                if modality in data_processors:
                    for per_processor in data_processors[modality]:
                        per_processor.load_tokenizer(path)

            # Only keep the modalities with non-empty processors.
            data_processors = {k: v for k, v in data_processors.items() if len(v) > 0}
        except:  # reconstruct the data processor in case something went wrong.
            data_processors = None

        learner._label_column = assets["label_column"]
        learner._problem_type = assets["problem_type"]
        learner._presets = assets["presets"]
        learner._best_score = assets["best_score"]
        learner._total_train_time = assets["total_train_time"]
        learner._eval_metric_name = assets["eval_metric_name"]
        with open(os.path.join(path, "eval_metric.pkl"), "rb") as fp:
            learner._eval_metric_func = pickle.load(fp)  # nosec B301
        learner._verbosity = verbosity
        learner._resume = resume
        learner._save_path = path  # in case the original exp dir is copied to somewhere else
        learner._pretrained_path = path
        learner._pretrained = assets["pretrained"]
        learner._fit_called = assets["fit_called"]
        learner._config = config
        learner._output_shape = assets["output_shape"]
        if "classes" in assets:
            learner._classes = assets["classes"]
        learner._column_types = assets["column_types"]
        learner._validation_metric_name = assets["validation_metric_name"]
        learner._df_preprocessor = df_preprocessor
        learner._data_processors = data_processors
        learner._minmax_mode = assets["minmax_mode"]

        return learner

    @classmethod
    def load(
        cls,
        path: str,
        resume: Optional[bool] = False,
        verbosity: Optional[int] = 3,
    ):
        """
        Load a learner object from a directory specified by `path`. The to-be-loaded learner
        can be completely or partially trained by .fit(). If a previous training has completed,
        it will load the checkpoint `model.ckpt`. Otherwise if a previous training accidentally
        collapses in the middle, it can load the `last.ckpt` checkpoint by setting `resume=True`.
        It also supports loading one specific checkpoint given its path.

        Parameters
        ----------
        path
            The directory to load the learner object.
        resume
            Whether to resume training from `last.ckpt`. This is useful when a training was accidentally
            broken during the middle and we want to resume the training from the last saved checkpoint.
        verbosity
            Verbosity levels range from 0 to 4 and control how much information is printed.
            Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).

        Returns
        -------
        The loaded learner object.
        """
        dir_path, ckpt_path = get_dir_ckpt_paths(path=path)

        assert os.path.isdir(dir_path), f"'{dir_path}' must be an existing directory."
        learner = cls(label="dummy_label")
        learner = cls._load_metadata(learner=learner, path=dir_path, resume=resume, verbosity=verbosity)
        peft = learner._config.optim.peft
        learner._model = create_fusion_model(
            config=learner._config,
            num_classes=learner._output_shape,
            classes=learner._classes if hasattr(learner, "_classes") else None,
            num_numerical_columns=len(learner._df_preprocessor.numerical_feature_names),
            num_categories=learner._df_preprocessor.categorical_num_categories,
            pretrained=False if not peft else True,  # set "pretrain=False" to prevent downloading online models
        )
        if learner._data_processors is None:
            learner._data_processors = create_fusion_data_processors(
                config=learner._config,
                model=learner._model,
            )
        load_path, ckpt_path = get_load_ckpt_paths(
            ckpt_path=ckpt_path,
            dir_path=dir_path,
            resume=resume,
        )
        learner._load_state_dict(
            path=load_path,
            strict=not peft,
        )
        learner._ckpt_path = ckpt_path
        loss_func = get_loss_func(
            problem_type=learner._problem_type,
            mixup_active=False,
            loss_func_name=learner._config.optim.loss_func,
            config=learner._config.optim,
            num_classes=learner._output_shape,
        )
        model_postprocess_fn = get_model_postprocess_fn(
            problem_type=learner._problem_type,
            loss_func=loss_func,
        )
        learner._model_postprocess_fn = model_postprocess_fn
        learner._config = learner.update_strategy_by_env(learner._config)

        return learner

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
        In binary classification, :class:`BaseLearner.predict_proba(as_multiclass=False)`
        returns the estimated probability that each row belongs to the positive class.
        Will print a warning and return None if called when `learner.problem_type != 'binary'`.

        Returns
        -------
        The positive class name in binary classification or None if the problem is not binary classification.
        """
        if self._problem_type != BINARY:
            logger.warning(
                f"Warning: Attempted to retrieve positive class label in a non-binary problem. "
                f"Positive class labels only exist in binary classification. "
                f"Returning None instead. The problem type is '{self._problem_type}'"
                f" but positive_class only exists for '{BINARY}'."
            )
            return None
        else:
            return self.class_labels[1]

    def fit_summary(self, verbosity=0, show_plot=False):
        """
        Output summary of information about models produced during `fit()`.

        Parameters
        ----------
        verbosity : int, default = 2
            Verbosity levels range from 0 to 4 and control how much information is printed.
            verbosity = 0 for no output printing.
            TODO: Higher levels correspond to more detailed print statements
        show_plot : bool, default = False
            If True, shows the model summary plot in browser when verbosity > 1.

        Returns
        -------
        Dict containing various detailed information.
        We do not recommend directly printing this dict as it may be very large.
        """
        if self._total_train_time is None:
            logging.info("There is no `best_score` or `total_train_time`. Have you called `predictor.fit()`?")
        else:
            logging.info(
                f"Here's the model summary:"
                f""
                f"The model achieved score '{self._best_score}' on the validation metric"
                f" '{self._validation_metric_name}'. "
                f"The total training time is {timedelta(seconds=self._total_train_time)}"
            )
        results = {
            f"val_{self._validation_metric_name}": self._best_score,
            "training_time": self._total_train_time,
        }
        return results

    def list_supported_models(self, pretrained=True):
        """
        List supported models for each problem_type to let users know
        options of checkpoint name to choose during fit().

        Parameters
        ----------
        pretrained : bool, default = True
            If True, only return the models with pretrained weights.
            If False, return all the models as long as there is model definition.

        Returns
        -------
        a list of model names
        """
        if self.problem_property and self.problem_property.is_classification:
            # FIXME (Need to list the supported models for each modality)
            return list_timm_models(pretrained=pretrained)
        else:
            raise ValueError(f"list_supported_models() is not available for problem type: {self._problem_type}")

    @staticmethod
    def update_strategy_by_env(config):
        """
        Set strategy to ddp_fork or ddp_notebook if an iterative env is detected.
        """
        assert config is not None
        if is_interactive_env() and not is_interactive_strategy(config.env.strategy):
            strs = list(config.env.strategy.partition("_find_unused_parameters"))
            strs[0] = "ddp_fork"
            config.env.strategy = "".join(strs)

        return config

    def set_num_gpus(self, num_gpus):
        assert isinstance(num_gpus, int)
        self._config.env.num_gpus = num_gpus

    def get_num_gpus(self):
        try:
            return self._config.env.num_gpus
        except:
            return None
