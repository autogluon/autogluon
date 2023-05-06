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
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import transformers
import yaml
from omegaconf import DictConfig, OmegaConf
from overrides import override
from packaging import version
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import CheckpointIO
from pytorch_lightning.strategies import Strategy
from timm.data.mixup import Mixup
from torch import nn
from torchmetrics import Metric

from autogluon.common.utils.log_utils import set_logger_verbosity, verbosity2loglevel
from autogluon.core.utils import default_holdout_frac, generate_train_test_split_combined
from autogluon.core.utils.loaders import load_pd

# from . import version as ag_version
from autogluon.multimodal.constants import (
    AUTOMM,
    AUTOMM_TUTORIAL_MODE,
    BBOX,
    BEST,
    BEST_K_MODELS_FILE,
    BINARY,
    COLUMN_FEATURES,
    DEEPSPEED_MIN_PL_VERSION,
    DEEPSPEED_MODULE,
    DEEPSPEED_OFFLOADING,
    DEEPSPEED_STRATEGY,
    DEPRECATED_ZERO_SHOT,
    DOCUMENT,
    FEATURE_EXTRACTION,
    FEATURES,
    FEW_SHOT,
    FEW_SHOT_TEXT_CLASSIFICATION,
    GREEDY_SOUP,
    IMAGE_BYTEARRAY,
    IMAGE_PATH,
    LABEL,
    LAST_CHECKPOINT,
    LOGITS,
    MAP,
    MASKS,
    MAX,
    MIN,
    MODEL_CHECKPOINT,
    MULTICLASS,
    NER,
    NER_RET,
    NUMERICAL,
    OBJECT_DETECTION,
    OCR_TEXT_DETECTION,
    OCR_TEXT_RECOGNITION,
    OVERALL_F1,
    RAY_TUNE_CHECKPOINT,
    REGRESSION,
    ROIS,
    SCORE,
    TEXT,
    TEXT_NER,
    UNIFORM_SOUP,
    XYWH,
    Y_PRED,
    Y_PRED_PROB,
    Y_TRUE,
    ZERO_SHOT_IMAGE_CLASSIFICATION,
)
from autogluon.multimodal.data.datamodule import BaseDataModule
from autogluon.multimodal.data.infer_types import (
    infer_column_types,
    infer_label_column_type_by_problem_type,
    infer_problem_type_output_shape,
    infer_rois_column_type,
    is_image_column,
)
from autogluon.multimodal.data.preprocess_dataframe import MultiModalFeaturePreprocessor
from autogluon.multimodal.matcher import MultiModalMatcher
from autogluon.multimodal.models.utils import get_model_postprocess_fn
from autogluon.multimodal.optimization.lit_distiller import DistillerLitModule
from autogluon.multimodal.optimization.lit_mmdet import MMDetLitModule
from autogluon.multimodal.optimization.lit_module import LitModule
from autogluon.multimodal.optimization.lit_ner import NerLitModule
from autogluon.multimodal.optimization.losses import RKDLoss
from autogluon.multimodal.optimization.utils import (
    get_loss_func,
    get_metric,
    get_norm_layer_param_names,
    get_trainable_params_efficient_finetune,
)
from autogluon.multimodal.problem_types import PROBLEM_TYPES_REG, ProblemTypeProperty
from autogluon.multimodal.utils import (
    AutoMMModelCheckpoint,
    AutoMMModelCheckpointIO,
    CustomUnpickler,
    DDPCacheWriter,
    ExportMixin,
    LogFilter,
    apply_log_filter,
    assign_feature_column_names,
    average_checkpoints,
    check_if_packages_installed,
    compute_grad_steps,
    compute_num_gpus,
    compute_score,
    convert_pred_to_xywh,
    create_fusion_data_processors,
    create_fusion_model,
    data_to_df,
    evaluate_coco,
    extract_from_output,
    filter_hyperparameters,
    get_available_devices,
    get_config,
    get_detection_classes,
    get_fit_complete_message,
    get_fit_start_message,
    get_local_pretrained_config_paths,
    get_minmax_mode,
    get_mixup,
    get_stopping_threshold,
    hyperparameter_tune,
    infer_dtypes_by_model_names,
    infer_metrics,
    infer_precision,
    infer_scarcity_mode_by_data_size,
    init_df_preprocessor,
    init_pretrained,
    list_timm_models,
    load_text_tokenizers,
    logits_to_prob,
    merge_bio_format,
    modify_duplicate_model_names,
    object_detection_data_to_df,
    predict,
    process_batch,
    save_pretrained_model_configs,
    save_result_df,
    save_text_tokenizers,
    select_model,
    setup_detection_train_tuning_data,
    setup_save_path,
    split_hyperparameters,
    tensor_to_ndarray,
    turn_on_off_feature_column_info,
    update_config_by_rules,
    update_hyperparameters,
    update_tabular_config_by_resources,
    upgrade_config,
)
from autogluon.multimodal.utils.log import get_fit_complete_message, get_fit_start_message
from .abstract_mm_learner import AbstractMMLearner

logger = logging.getLogger(__name__)


class DefaultLearner(AbstractMMLearner):
    def __init__(
        self,
        problem_type: str,
        label: Optional[str] = None,
        # column_types: Optional[dict] = None,  # this is now inferred in learner
        # query: Optional[Union[str, List[str]]] = None,
        # response: Optional[Union[str, List[str]]] = None,
        # match_label: Optional[Union[int, str]] = None,
        # pipeline: Optional[str] = None,
        presets: Optional[str] = None,
        eval_metric: Optional[str] = None,  # this is now inferred in learner
        path: Optional[str] = None,
        verbosity: Optional[int] = 2,
        num_classes: Optional[int] = None,
        classes: Optional[list] = None,
        enable_progress_bar: Optional[bool] = True,
    ):
        super().__init__()

        self._label_column: Optional[str] = label
        self._problem_type: Optional[str] = problem_type
        # self._presets = presets.lower() if presets else None  # NOTE: This will be handled by predictor.
        self._eval_metric_name: Optional[str] = eval_metric
        self._save_path = path
        self._verbosity = verbosity
        self._output_shape: Optional[int] = num_classes
        self._classes = classes  # TODO: To be refactored into detection learner only
        self._presets = presets

        self._ckpt_path: Optional[str] = None
        self._config: Optional[DictConfig] = None
        self._column_types: Optional[dict] = None  # this is now inferred in learner
        self._data_processors: Optional[List] = None  # initialize it as None.
        self._df_preprocessor: Optional[MultiModalFeaturePreprocessor] = None  # initialize it as None.
        self._model_postprocess_fn: Optional[Callable] = None
        self._model: Optional[nn.Module] = None
        self._pretrained_path: Optional[str] = None
        self._resume = False  # now learner handles this as well
        self._validation_metric_name: Optional[str] = None

        # self._warn_if_exist = warn_if_exist  # unused
        self._enable_progress_bar = enable_progress_bar
        # self._init_scratch = init_scratch  # NOTE: is this for predictor to handle?
        # self._sample_data_path = sample_data_path  # this is for detection problem only
        self._fit_called = False  # While using ddp, after fit called, we can only use single gpu.

        # Summary statistics used in fit summary. TODO: wrap it in a class.
        # self._total_train_time = None
        self._best_score = None

    @override
    def path(self) -> Union[str, None]:
        return self._save_path

    @property
    @override
    def label(self) -> Union[str, None]:
        return self._label_column

    @property
    @override
    def problem_type(self) -> Union[str, None]:
        return self._problem_type

    @property
    @override
    def problem_property(self) -> Union[ProblemTypeProperty, None]:
        if self._problem_type is None:
            return None
        else:
            return PROBLEM_TYPES_REG.get(self._problem_type)

    @property
    @override
    def class_labels(self) -> Union[list, None]:
        return self._classes

    @property
    @override
    def positive_class(self) -> Union[int, None]:
        return None

    def _prepare_fit_args(
        self,
        train_data: Union[pd.DataFrame, str],
        presets: Optional[str] = None,
        config: Optional[dict] = None,
        tuning_data: Optional[Union[pd.DataFrame, str]] = None,
        max_num_tuning_data: Optional[int] = None,  # TODO: Remove since this is for detection
        # TODO: Remove since this is for matching
        id_mappings: Optional[Union[Dict[str, Dict], Dict[str, pd.Series]]] = None,
        time_limit: Optional[int] = None,
        save_path: Optional[str] = None,
        hyperparameters: Optional[Union[str, Dict, List[str]]] = None,
        column_types: Optional[dict] = None,
        holdout_frac: Optional[float] = None,
        teacher_predictor: Union[str, MultiModalPredictor] = None,
        seed: Optional[int] = 0,
        standalone: Optional[bool] = True,
        hyperparameter_tune_kwargs: Optional[dict] = None,
        clean_ckpts: Optional[bool] = True,
    ):
        """Prepare arguments for fit().
        Parameters
        ----------
        train_data
            A dataframe containing training data.
        presets
            Presets regarding model quality, e.g., best_quality, high_quality, and medium_quality.
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
        id_mappings
             Id-to-content mappings. The contents can be text, image, etc.
             This is used when the dataframe contains the query/response identifiers instead of their contents.
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
            Defaults to 0
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
        clean_ckpts
            Whether to clean the checkpoints of each validation step after training.

        Returns
        -------
        An "MultiModalPredictor" object (itself).
        """
        # 1. some sanity checks
        fit_called = self._fit_called  # used in current function
        self._fit_called = True
        pl.seed_everything(seed, workers=True)

        if self._problem_type and not self.problem_property.support_fit:
            raise RuntimeError(
                f"The problem_type='{self._problem_type}' does not support `predictor.fit()`. "
                f"You may try to use `predictor.predict()` or `predictor.evaluate()`."
            )

        if time_limit is not None:
            time_limit = timedelta(seconds=time_limit)
        training_start = time.time()

        # 2. Route to Matcher if applicable. This can be an example of how a learner would work.
        # TODO: Predictor responsible for creating a new learner for matching.
        # if self._matcher:
        #     self._matcher.fit(
        #         train_data=train_data,
        #         tuning_data=tuning_data,
        #         id_mappings=id_mappings,
        #         time_limit=time_limit,
        #         presets=presets,
        #         hyperparameters=hyperparameters,
        #         column_types=column_types,
        #         holdout_frac=holdout_frac,
        #         save_path=save_path,
        #         hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
        #         seed=seed,
        #     )
        #     return self

        # 3. Setup and split data
        # TODO: Move to detection learner's _setup_train_tuning_data() method
        # if self._problem_type == OBJECT_DETECTION:
        #     train_data, tuning_data = setup_detection_train_tuning_data(
        #         self, max_num_tuning_data, seed, train_data, tuning_data
        #     )
        # 6. setup save path
        self._setup_save_path(save_path, fit_called)

        train_data, tuning_data = self._setup_train_tuning_data(
            train_data=train_data, tuning_data=tuning_data, holdout_frac=holdout_frac, seed=seed
        )

        # 4. setup presets. Silently ignore user input if self._presets already exists
        if self._presets is not None:
            # FIXME: Silently ignoring user input, there should be a warning
            presets = self._presets
        else:
            self._presets = presets

        # 5. setup config. Silently ignore user input if self._config already exists
        if self._config is not None:  # continuous training
            # FIXME: Silently ignoring user input, there should be a warning
            config = self._config
        else:
            self._config = config

        # 7. setup problem types.
        # TODO: Remove. The problem type is inferred from predictor, and should've been assigned in __init__
        # self._problem_type = self._infer_problem_type(train_data=train_data, column_types=column_types)

        # 8. infer column types
        self._column_types = self._infer_column_types(
            train_data=train_data, tuning_data=tuning_data, column_types=column_types
        )
        # 10. ignore user input if self._column_type already exists.
        # TODO: Merged with _infer_column_types()

        # 9. infer output shape
        # FIXME: separate infer problem_type with output_shape, should be logically distinct
        self._output_shape = self._infer_output_shape(train_data=train_data)
        # 11. check for output shape consistency. Raise error if inconsistent for obj. det.
        # TODO: Merged with _infer_output_shape()
        # TODO: Delete this check for object detection
        # if self._problem_type != OBJECT_DETECTION:
        #     if self._output_shape is not None and output_shape is not None:
        #         assert self._output_shape == output_shape, (
        #             f"Inferred output shape {output_shape} is different from " f"the previous {self._output_shape}"
        #         )
        #     else:
        #         self._output_shape = output_shape

        # Determine data scarcity mode, i.e. a few-shot scenario
        self._check_if_fewshot(train_data, presets)

        logger.debug(f"self._column_types: {self._column_types}")
        logger.debug(f"image columns: {[k for k, v in self._column_types.items() if v == 'image_path']}")

        # 12. setup validation metric names
        self._validation_metric_name, self._eval_metric_name = self._infer_metric_names()

        # 13. get minmax mode. FIXME: Can be merged with step 0 at the top
        # TODO: remove precompute to save an argument. Move to .fit() and compute when needed.
        minmax_mode = get_minmax_mode(self._validation_metric_name)

        # set attributes for saving and prediction
        # self._eval_metric_name = eval_metric_name  # In case eval_metric isn't provided in __init__().
        # self._validation_metric_name = validation_metric_name

        # 14. setup hyperparameters and hpo hyperparameters
        hyperparameters, advanced_hyperparameters, hyperparameter_tune_kwargs = self._setup_hyperparameters(
            hyperparameters=hyperparameters,
            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
            teacher_predictor=teacher_predictor,
        )

        hpo_mode = self._is_hpo_mode(hyperparameter_tune_kwargs)
        # 15. filter out the hyperparameters that have no effect for HPO.
        if hpo_mode:
            hyperparameters = filter_hyperparameters(
                hyperparameters=hyperparameters,
                column_types=self._column_types,
                config=config,
                fit_called=fit_called,
            )

        _fit_args = dict(
            train_df=train_data,
            val_df=tuning_data,
            validation_metric_name=self._validation_metric_name,  # TODO: Remove this args since _validation_metric_name is already set in self.
            minmax_mode=minmax_mode,
            max_time=time_limit,
            save_path=self._save_path,  # TODO: Remove this args since _save_path is already set in self.
            ckpt_path=None if hpo_mode else self._ckpt_path,
            resume=False if hpo_mode else self._resume,
            enable_progress_bar=False if hpo_mode else self._enable_progress_bar,
            presets=presets,
            hyperparameters=hyperparameters,
            advanced_hyperparameters=advanced_hyperparameters,
            teacher_predictor=teacher_predictor,
            standalone=standalone,
            hpo_mode=hpo_mode,  # skip average checkpoint if in hpo mode
            clean_ckpts=clean_ckpts,
        )

        return _fit_args

    def _is_hpo_mode(self, hyperparameter_tune_kwargs):
        hpo_mode = True if hyperparameter_tune_kwargs else False
        return hpo_mode

    def _setup_hyperparameters(self, hyperparameters, hyperparameter_tune_kwargs, teacher_predictor):
        hyperparameters, hyperparameter_tune_kwargs = update_hyperparameters(
            problem_type=self._problem_type,
            presets=self._presets,
            provided_hyperparameters=hyperparameters,
            provided_hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
            teacher_predictor=teacher_predictor,
        )
        # split out the hyperparameters whose values are complex objects
        hyperparameters, advanced_hyperparameters = split_hyperparameters(hyperparameters)
        return (
            hyperparameters,
            advanced_hyperparameters,
            hyperparameter_tune_kwargs,
        )

    def _setup_config(self):
        if self._config is not None:  # continuous training
            # FIXME: Silently ignoring user input, there should be a warning
            config = self._config
        return config

    def _setup_presets(self):
        if self._presets is not None:
            # FIXME: Silently ignoring user input, there should be a warning
            presets = self._presets
        else:
            self._presets = presets
        return presets

    def _setup_save_path(self, save_path, fit_called):
        self._save_path = setup_save_path(
            resume=self._resume,
            old_save_path=self._save_path,
            proposed_save_path=save_path,
            raise_if_exist=True,
            warn_if_exist=False,
            fit_called=fit_called,
        )

    def _infer_metric_names(self):
        if self._validation_metric_name is None or self._eval_metric_name is None:
            validation_metric_name, eval_metric_name = infer_metrics(
                problem_type=self._problem_type,
                eval_metric_name=self._eval_metric_name,
                validation_metric_name=self._validation_metric_name,
            )
        else:
            validation_metric_name = self._validation_metric_name
            eval_metric_name = self._eval_metric_name
        return validation_metric_name, eval_metric_name

    def _check_if_fewshot(self, train_data, presets):
        scarcity_mode = infer_scarcity_mode_by_data_size(
            df_train=train_data, scarcity_threshold=50
        )  # Add as separate hyperparameter somewhere?
        if scarcity_mode == FEW_SHOT and (not presets or FEW_SHOT not in presets):  # TODO: check for data  type
            logger.info(
                f"Detected data scarcity. Consider running using the preset '{FEW_SHOT_TEXT_CLASSIFICATION}' for better performance."
            )

    def _infer_output_shape(self, train_data):
        _, output_shape = infer_problem_type_output_shape(
            label_column=self._label_column,
            column_types=self._column_types,
            data=train_data,
            provided_problem_type=self._problem_type,
        )

        # TODO: Delete this check for object detection
        # if self._problem_type != OBJECT_DETECTION:
        if self._output_shape is not None and output_shape is not None:
            assert self._output_shape == output_shape, (
                f"Inferred output shape {output_shape} is different from " f"the previous {self._output_shape}"
            )
        # else:
        #     self._output_shape = output_shape

        return output_shape

    def _infer_column_types(
        self, train_data: pd.DataFrame, tuning_data: pd.DataFrame = None, column_types: dict = None
    ) -> dict:
        column_types = infer_column_types(
            data=train_data,
            label_columns=self._label_column,
            provided_column_types=column_types,
            valid_data=tuning_data,
            problem_type=self._problem_type,
        )
        column_types = infer_label_column_type_by_problem_type(
            column_types=column_types,
            label_columns=self._label_column,
            problem_type=self._problem_type,
            data=train_data,
            valid_data=tuning_data,
        )
        if self._column_types is not None and self._column_types != column_types:
            warnings.warn(
                f"Inferred column types {column_types} are inconsistent with "
                f"the previous {self._column_types}. "
                f"New columns will not be used in the current training."
            )
            # use previous column types to avoid inconsistency with previous numerical mlp and categorical mlp
            column_types = self._column_types

        return column_types

    def _setup_train_tuning_data(
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
            train_data, tuning_data = self._split_train_tuning(
                data=train_data, holdout_frac=holdout_frac, random_state=seed
            )

        return train_data, tuning_data

    def _split_train_tuning(
        self, data: pd.DataFrame, holdout_frac: float = None, random_state: int = 0
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits `data` into `train_data` and `tuning_data`.
        If the problem_type is one of ['binary', 'multiclass']:
            The split will be done with stratification on the label column.
            Will guarantee at least 1 sample of every class in `data` will be present in `train_data`.
                If only 1 sample of a class exists, it will always be put in `train_data` and not `tuning_data`.

        Parameters
        ----------
        data : pd.DataFrame
            The data to be split
        holdout_frac : float, default = None
            The ratio of data to use as validation.
            If 0.2, 20% of the data will be used for validation, and 80% for training.
            If None, the ratio is automatically determined,
            ranging from 0.2 for small row count to 0.01 for large row count.
        random_state : int, default = 0
            The random state to use when splitting the data, to make the splitting process deterministic.
            If None, a random value is used.

        Returns
        -------
        Tuple of (train_data, tuning_data) of the split `data`
        """
        if holdout_frac is None:
            holdout_frac = default_holdout_frac(num_train_rows=len(data), hyperparameter_tune=False)

        # TODO: Hack since the recognized problem types are only binary, multiclass, and regression
        #  Problem types used for purpose of stratification, so regression = no stratification
        if self._problem_type in [BINARY, MULTICLASS]:
            problem_type_for_split = self._problem_type
        else:
            problem_type_for_split = REGRESSION

        train_data, tuning_data = generate_train_test_split_combined(
            data=data,
            label=self.label,
            test_size=holdout_frac,
            problem_type=problem_type_for_split,
            random_state=random_state,
        )
        return train_data, tuning_data

    @override
    def fit(
        self,
        train_data: Union[pd.DataFrame, str],
        presets: Optional[str] = None,
        config: Optional[dict] = None,
        tuning_data: Optional[Union[pd.DataFrame, str]] = None,
        max_num_tuning_data: Optional[int] = None,  # TODO: Remove since this is for detection
        # TODO: Remove since this is for matching
        id_mappings: Optional[Union[Dict[str, Dict], Dict[str, pd.Series]]] = None,
        time_limit: Optional[int] = None,
        save_path: Optional[str] = None,
        hyperparameters: Optional[Union[str, Dict, List[str]]] = None,
        column_types: Optional[dict] = None,
        holdout_frac: Optional[float] = None,
        teacher_predictor: Union[str, DefaultLearner] = None,
        seed: Optional[int] = 0,
        standalone: Optional[bool] = True,
        hyperparameter_tune_kwargs: Optional[dict] = None,
        clean_ckpts: Optional[bool] = True,
    ):
        _fit_args = self._prepare_fit_args(
            train_data=train_data,
            presets=presets,
            config=config,
            tuning_data=tuning_data,
            max_num_tuning_data=max_num_tuning_data,
            id_mappings=id_mappings,
            time_limit=time_limit,
            save_path=save_path,
            hyperparameters=hyperparameters,
            column_types=column_types,
            holdout_frac=holdout_frac,
            teacher_predictor=teacher_predictor,
            seed=seed,
            standalone=standalone,
            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
            clean_ckpts=clean_ckpts,
        )
        self._run_fit(**_fit_args)

    def _run_fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        validation_metric_name: str,
        minmax_mode: str,  # this is determined solely on validation_metric_name
        save_path: str,
        ckpt_path: str,  # these two are synced
        resume: bool,  # these two are synced
        max_time: Optional[timedelta] = None,
        enable_progress_bar: Optional[bool] = False,  # can be inferred?
        presets: Optional[str] = None,
        config: Optional[dict] = None,
        hyperparameters: Optional[Union[str, Dict, List[str]]] = None,
        advanced_hyperparameters: Optional[Dict] = None,
        teacher_predictor: Union[str, DefaultLearner] = None,
        hpo_mode: bool = False,
        standalone: bool = True,
        clean_ckpts: bool = True,
        **hpo_kwargs,
    ):
        # TODO(?) We should have a separate "_pre_training_event()" for logging messages.
        logger.info(get_fit_start_message(save_path, validation_metric_name))

        ### SETUP Environments ###
        # self._fit_called = True
        # 1-3, 22. get config.
        # TODO: Create another PR to isolate df_preprocessor from self._setup_environment().
        # Create another method self._create_df_preprocessor(), self._get_data_processors(), and _create_model()
        config, df_preprocessor, grad_steps, strategy, use_ray_lightning = self._setup_environment(
            train_df=train_df,
            config=config,
            presets=presets,
            hyperparameters=hyperparameters,
            teacher_predictor=teacher_predictor,
            hpo_mode=hpo_mode,
            **hpo_kwargs,
        )
        self._config = config
        self._df_preprocessor = df_preprocessor

        # 4-5. if NER, update output shape. Otherwise create model. TODO: This can be refactored into the NER Learner
        self._model = self._create_model()
        # self._model = model

        # 6. get trainable layer params for efficient finetuning only?
        trainable_param_names = self._get_trainable_params()

        # 7. get data_processors.
        self._data_processors = self._get_data_processors(advanced_hyperparameters=advanced_hyperparameters)
        # self._data_processors = data_processors

        # 8. infer positive labels (NOTE: This is deleted in the latest main branch)

        # 9. setup validation metric
        validation_metric, custom_metric_func = self._setup_validation_metric(
            validation_metric_name=validation_metric_name
        )

        # 10. setup mix up if applicable
        mixup_active, mixup_fn = self._setup_mixup()

        # 11. get loss function, NOTE: This can be refactored into Learner class
        loss_func = get_loss_func(
            problem_type=self._problem_type,
            mixup_active=mixup_active,
            loss_func_name=OmegaConf.select(self._config, "optimization.loss_function"),
            config=self._config.optimization,
        )

        # 12. get post process function
        self._model_postprocess_fn = get_model_postprocess_fn(
            problem_type=self._problem_type,
            loss_func=loss_func,
        )
        # self._model_postprocess_fn = model_postprocess_fn

        # 13. assign properties, and use self properties afterward whenever possible
        # self._config = config
        # self._df_preprocessor = df_preprocessor
        # self._data_processors = data_processors
        # self._model = model
        # self._model_postprocess_fn = model_postprocess_fn

        # TODO: complete this later
        # 14. if times up. return final model
        if max_time == timedelta(seconds=0):
            self._top_k_average(
                model=self._model,
                save_path=save_path,
                minmax_mode=minmax_mode,
                is_distill=False,
                top_k_average_method=self._config.optimization.top_k_average_method,
                val_df=val_df,
                validation_metric_name=validation_metric_name,
                strict_loading=not trainable_param_names,
                standalone=standalone,
                clean_ckpts=clean_ckpts,
            )

            return self

        # By Default, no distillation. So ignore for now.
        # TODO: refactor this into a separate function called self.distill()
        # which is invoked by self.fit() if teacher_predictor is not None
        # 15. setup distillation.
        # need to assign the above attributes before setting up distillation
        # if teacher_predictor is not None:
        #     (
        #         teacher_model,
        #         critics,
        #         baseline_funcs,
        #         soft_label_loss_func,
        #         softmax_regression_loss_func,
        #         output_feature_adaptor,
        #         output_feature_loss_func,
        #         rkd_loss_func,
        #         teacher_df_preprocessor,
        #         teacher_data_processors,
        #     ) = self._setup_distillation(
        #         teacher_predictor=teacher_predictor,
        #     )
        # else:
        #     (
        #         teacher_model,
        #         critics,
        #         baseline_funcs,
        #         soft_label_loss_func,
        #         softmax_regression_loss_func,
        #         output_feature_adaptor,
        #         output_feature_loss_func,
        #         rkd_loss_func,
        #         teacher_df_preprocessor,
        #         teacher_data_processors,
        #     ) = (None, None, None, None, None, None, None, None, None, None)

        # if teacher_df_preprocessor is not None:
        #     df_preprocessor = [df_preprocessor, teacher_df_preprocessor]
        # if teacher_data_processors is not None:
        #     data_processors = [data_processors, teacher_data_processors]

        # TODO: Refactor this into detection learner
        val_use_training_mode = (self._problem_type == OBJECT_DETECTION) and (validation_metric_name != MAP)

        # 16. setup pl training data module
        train_dm = self._get_train_data_module(
            train_df=train_df,
            val_df=val_df,
            val_use_training_mode=val_use_training_mode,
        )

        # 17. setup optim args
        optimization_kwargs = self._get_optim_kwargs()
        # 18. setup metrics args
        metrics_kwargs = self._get_metrics_kwargs(
            validation_metric_name=validation_metric_name,
            validation_metric=validation_metric,
            custom_metric_func=custom_metric_func,
        )

        # By default, we use the default task type a.k.a. LitModule
        # TODO: Refactor this into sub-learners
        # TODO: i.e. define function self._setup_task_lightning_module() -> pl.LightningModule
        # 19. select the correct pl module for a given problem type.
        # NOTE: The task creation and all related variables/functions should be refactored into each Learner class
        # is_distill = teacher_model is not None
        # if is_distill:
        #     output_feature_loss_weight = OmegaConf.select(
        #         self._config, "distiller.output_feature_loss_weight", default=0.0
        #     )
        #     softmax_regression_weight = OmegaConf.select(
        #         self._config, "distiller.softmax_regression_weight", default=0.0
        #     )
        #     use_raw_features = OmegaConf.select(self._config, "distiller.use_raw_features", default=False)
        #     task = DistillerLitModule(
        #         student_model=model,
        #         teacher_model=teacher_model,
        #         matches=config.distiller.matches,
        #         critics=critics,
        #         baseline_funcs=baseline_funcs,
        #         hard_label_weight=config.distiller.hard_label_weight,
        #         soft_label_weight=config.distiller.soft_label_weight,
        #         softmax_regression_weight=softmax_regression_weight,
        #         temperature=config.distiller.temperature,
        #         output_feature_loss_weight=output_feature_loss_weight,
        #         hard_label_loss_func=loss_func,
        #         soft_label_loss_func=soft_label_loss_func,
        #         softmax_regression_loss_func=softmax_regression_loss_func,
        #         output_feature_adaptor=output_feature_adaptor,
        #         output_feature_loss_func=output_feature_loss_func,
        #         rkd_loss_func=rkd_loss_func,
        #         **metrics_kwargs,
        #         **optimization_kwargs,
        #     )
        # elif self._problem_type == NER:
        #     task = NerLitModule(
        #         model=model,
        #         loss_func=loss_func,
        #         efficient_finetune=OmegaConf.select(config, "optimization.efficient_finetune"),
        #         mixup_fn=mixup_fn,
        #         mixup_off_epoch=OmegaConf.select(config, "data.mixup.turn_off_epoch"),
        #         model_postprocess_fn=model_postprocess_fn,
        #         trainable_param_names=trainable_param_names,
        #         **metrics_kwargs,
        #         **optimization_kwargs,
        #     )
        # elif self._problem_type == OBJECT_DETECTION:
        #     task = MMDetLitModule(
        #         model=model,
        #         **metrics_kwargs,
        #         **optimization_kwargs,
        #     )
        # else:
        #     task = LitModule(
        #         model=model,
        #         loss_func=loss_func,
        #         efficient_finetune=OmegaConf.select(config, "optimization.efficient_finetune"),
        #         mixup_fn=mixup_fn,
        #         mixup_off_epoch=OmegaConf.select(config, "data.mixup.turn_off_epoch"),
        #         model_postprocess_fn=model_postprocess_fn,
        #         trainable_param_names=trainable_param_names,
        #         skip_final_val=OmegaConf.select(config, "optimization.skip_final_val", default=False),
        #         **metrics_kwargs,
        #         **optimization_kwargs,
        #     )

        task = self._setup_task_lightning_module(
            trainable_param_names=trainable_param_names,
            mixup_fn=mixup_fn,
            loss_func=loss_func,
            optimization_kwargs=optimization_kwargs,
            metrics_kwargs=metrics_kwargs,
        )

        # 20. Setup call backs
        checkpoint_callback = self._create_checkpoint_callback(minmax_mode=minmax_mode, save_path=save_path, task=task)
        early_stopping_callback = self._create_early_stop_callback(
            validation_metric_name=validation_metric_name, minmax_mode=minmax_mode, task=task
        )
        lr_callback = self._create_lr_callback()
        model_summary = pl.callbacks.ModelSummary(max_depth=1)
        callbacks = [
            checkpoint_callback,
            early_stopping_callback,
            lr_callback,
            model_summary,
        ]

        # 21. for hpo with ray
        # TODO: Leave this out for now. We will add this back later
        # use_ray_lightning = "_ray_lightning_plugin" in hpo_kwargs
        # if hpo_mode:
        #     if use_ray_lightning:
        #         from ray_lightning.tune import TuneReportCheckpointCallback
        #     else:
        #         from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
        #     tune_report_callback = TuneReportCheckpointCallback(
        #         {f"{task.validation_metric_name}": f"{task.validation_metric_name}"},
        #         filename=RAY_TUNE_CHECKPOINT,
        #     )
        #     callbacks = [
        #         tune_report_callback,
        #         early_stopping_callback,
        #         lr_callback,
        #         model_summary,
        #     ]

        custom_checkpoint_plugin = AutoMMModelCheckpointIO(
            trainable_param_names=trainable_param_names,
            model_name_to_id=self._model.name_to_id,
        )

        tb_logger = pl.loggers.TensorBoardLogger(
            save_dir=save_path,
            name="",
            version="",
        )

        ## Moved to the begining to join the setup steps for config
        # # 22. training environment-dependent configs
        # num_gpus = compute_num_gpus(config_num_gpus=config.env.num_gpus, strategy=config.env.strategy)

        # precision = infer_precision(num_gpus=num_gpus, precision=config.env.precision)

        # if num_gpus == 0:  # CPU only training
        #     grad_steps = max(
        #         config.env.batch_size // (config.env.per_gpu_batch_size * config.env.num_nodes),
        #         1,
        #     )
        # else:
        #     grad_steps = max(
        #         config.env.batch_size // (config.env.per_gpu_batch_size * num_gpus * config.env.num_nodes),
        #         1,
        #     )

        # if not hpo_mode:
        #     if num_gpus <= 1:
        #         if config.env.strategy == DEEPSPEED_OFFLOADING:  # Offloading currently only tested for single GPU
        #             assert version.parse(pl.__version__) >= version.parse(
        #                 DEEPSPEED_MIN_PL_VERSION
        #             ), f"For DeepSpeed Offloading to work reliably you need at least pytorch-lightning version {DEEPSPEED_MIN_PL_VERSION}, however, found {pl.__version__}. Please update your pytorch-lightning version."
        #             from .optimization.deepspeed import CustomDeepSpeedStrategy

        #             strategy = CustomDeepSpeedStrategy(
        #                 stage=3,
        #                 offload_optimizer=True,
        #                 offload_parameters=False,
        #                 allgather_bucket_size=config.env.deepspeed_allgather_size,
        #                 reduce_bucket_size=config.env.deepspeed_allreduce_size,
        #             )
        #         else:
        #             strategy = None
        #     else:
        #         strategy = config.env.strategy
        # else:
        #     # we don't support running each trial in parallel without ray lightning
        #     if use_ray_lightning:
        #         strategy = hpo_kwargs.get("_ray_lightning_plugin")
        #     else:
        #         strategy = None
        #         num_gpus = min(num_gpus, 1)

        # config.env.num_gpus = num_gpus
        # config.env.precision = precision
        # config.env.strategy = strategy if not config.env.strategy == DEEPSPEED_OFFLOADING else DEEPSPEED_OFFLOADING
        # self._config = config

        ## TODO: Complete the save later. IGNORE THIS FOR NOW
        # save artifacts for the current running, except for model checkpoint, which will be saved in trainer
        # self.save(save_path, standalone=standalone)

        blacklist_msgs = ["already configured with model summary"]
        log_filter = LogFilter(blacklist_msgs)
        # 23. Declare pl.Trainer
        with apply_log_filter(log_filter):
            trainer = self._get_trainer(
                enable_progress_bar=enable_progress_bar,
                grad_steps=grad_steps,
                strategy=strategy,
                use_ray_lightning=use_ray_lightning,
                callbacks=callbacks,
                custom_checkpoint_plugin=custom_checkpoint_plugin,
                tb_logger=tb_logger,
                max_time=max_time,
            )

        # 24. pl.Trainer.fit()
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

        # 25. execute call bakcs at training finish
        if trainer.global_rank == 0:
            # We do not perform averaging checkpoint in the case of hpo for each trial
            # We only averaging the checkpoint of the best trial in the end in the master process
            if not hpo_mode:
                self._top_k_average(
                    model=self._model,
                    save_path=save_path,
                    minmax_mode=minmax_mode,
                    is_distill=False,  # By default, we do not do distill. If distill, we will do it in the distill learner
                    top_k_average_method=config.optimization.top_k_average_method,
                    val_df=val_df,
                    validation_metric_name=validation_metric_name,
                    strategy=strategy,
                    strict_loading=not trainable_param_names,
                    # Not strict loading if using parameter-efficient finetuning
                    standalone=standalone,
                    clean_ckpts=clean_ckpts,
                )
            self._best_score = trainer.callback_metrics[f"val_{validation_metric_name}"].item()
        else:
            sys.exit(f"Training finished, exit the process with global_rank={trainer.global_rank}...")

    def _get_trainer(
        self,
        enable_progress_bar: bool,
        grad_steps: int,
        strategy: Optional[Union[Strategy, str]],
        use_ray_lightning: bool,
        callbacks: Optional[List[Callback]],
        custom_checkpoint_plugin: CheckpointIO,
        tb_logger: TensorBoardLogger,
        max_time: Optional[timedelta] = None,
    ) -> pl.Trainer:
        """Create the pytorch_lightning trainer for training

        Parameters
        ----------
        enable_progress_bar (bool): enable progress bar to display in console
        grad_steps (int): number of gradient steps to take
        strategy (Optional[Union[Strategy, str]]): data parallel strategy
        use_ray_lightning (bool): whether to use ray lightning plugin
        callbacks (Optional[List[Callback]]): pytorch lightning callbacks during training loop
        custom_checkpoint_plugin (CheckpointIO): custome pytorch lightning checkpoint plugin that customize the checkpoint saving
        tb_logger (TensorBoardLogger): tensorboard logger
        max_time (Optional[timedelta], optional): maximum trainig. Defaults to None.

        Returns
        -------
        pl.Trainer: the pytorch lightning trainer object for training
        """
        assert (
            self._config is not None
        ), "self._config is None. You must setup self._config before calling _get_trainer()"
        accelerator = "gpu" if self._config.env.num_gpus > 0 else None
        available_devices = get_available_devices(
            num_gpus=self._config.env.num_gpus,
            auto_select_gpus=self._config.env.auto_select_gpus,
            use_ray_lightning=use_ray_lightning,
        )
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=available_devices,
            num_nodes=self._config.env.num_nodes,
            precision=self._config.env.precision,
            strategy=strategy,
            benchmark=False,
            deterministic=self._config.env.deterministic,
            max_epochs=self._config.optimization.max_epochs,
            max_steps=self._config.optimization.max_steps,
            max_time=max_time,
            callbacks=callbacks,
            logger=tb_logger,
            gradient_clip_val=OmegaConf.select(self._config, "optimization.gradient_clip_val", default=1),
            gradient_clip_algorithm=OmegaConf.select(
                self._config, "optimization.gradient_clip_algorithm", default="norm"
            ),
            accumulate_grad_batches=grad_steps,
            log_every_n_steps=OmegaConf.select(self._config, "optimization.log_every_n_steps", default=10),
            enable_progress_bar=enable_progress_bar,
            fast_dev_run=self._config.env.fast_dev_run,
            track_grad_norm=OmegaConf.select(self._config, "optimization.track_grad_norm", default=-1),
            val_check_interval=self._config.optimization.val_check_interval,
            check_val_every_n_epoch=OmegaConf.select(self._config, "optimization.check_val_every_n_epoch", default=1),
            plugins=[custom_checkpoint_plugin],
        )

        return trainer

    @override
    def predict(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        candidate_data: Optional[Union[pd.DataFrame, dict, list]] = None,
        id_mappings: Optional[Union[Dict[str, Dict], Dict[str, pd.Series]]] = None,
        as_pandas: Optional[bool] = None,
        realtime: Optional[bool] = None,
        save_results: Optional[bool] = None,
        **kwargs,
    ):
        self._verify_inference_ready()

        # TODO: refactor into matching learner. Leave it as is for now in predictor.
        # if self._matcher:
        #     return self._matcher.predict(
        #         data=data,
        #         id_mappings=id_mappings,
        #         as_pandas=as_pandas,
        #         realtime=realtime,
        #     )

        # TODO: refactor into detection learner.
        # i.e. define self._transform_data_for_predict(data) -> pd.DataFrame
        # if self._problem_type == OBJECT_DETECTION:
        # data = object_detection_data_to_df(data)

        #     if self._label_column not in data:
        #         self._label_column = None

        # TODO: refactor into sub-learners
        # i.e. define self._get_prediction_ret_type() -> str
        # or just init a mapping in this default_learner
        # I think defining a function makes more sense.

        # if self._problem_type == OBJECT_DETECTION or self._problem_type == OCR_TEXT_DETECTION:
        #     ret_type = BBOX
        # elif self._problem_type == OCR_TEXT_RECOGNITION:
        #     ret_type = [TEXT, SCORE]
        # else:
        #     ret_type = LOGITS
        ret_type = LOGITS

        # if self._problem_type == NER:
        #     ret_type = NER_RET

        # TODO: This is a matching problem type, except the self._matcher is None
        if candidate_data:
            # I believe this only apply to a certain problem types
            # i.e. obviously this wouldn't make sense for OBJECT_DETECTION, OCR, etc.
            # From what I understand, this is more likely using models such CLIP, SWIN to extract embeddings
            # This is problem_type agnostic, but depends on what model is used.
            pred = self._match_queries_and_candidates(
                query_data=data,
                candidate_data=candidate_data,
                return_prob=False,
            )
        else:
            outputs = predict(
                predictor=self,
                data=data,
                requires_label=False,
                realtime=realtime,
            )

            # TODO: refactor into sub-learners
            # i.e. define function self._extract_from_output(outputs, ret_type) -> logits
            # if self._problem_type == OCR_TEXT_RECOGNITION:
            #     logits = []
            #     for r_type in ret_type:
            #         logits.append(extract_from_output(outputs=outputs, ret_type=r_type))
            # else:
            #     logits = extract_from_output(outputs=outputs, ret_type=ret_type)
            logits = extract_from_output(outputs=outputs, ret_type=ret_type)

            if self._df_preprocessor:
                if ret_type == BBOX:
                    pred = logits
                else:
                    pred = self._df_preprocessor.transform_prediction(
                        y_pred=logits,
                    )
            else:
                if isinstance(logits, (torch.Tensor, np.ndarray)) and logits.ndim == 2:
                    pred = logits.argmax(axis=1)
                else:
                    pred = logits

            # TODO: refactor into sub-learners
            # i.e. define function self._postprocess_pred(pred, **kwargs) -> pred
            # if self._problem_type == NER:
            #     pred = merge_bio_format(data[self._df_preprocessor.ner_feature_names[0]], pred)

            # if self._problem_type == OBJECT_DETECTION:
            #     if self._model.output_bbox_format == XYWH:
            #         pred = convert_pred_to_xywh(pred)

        # TODO: refactor into sub-learners
        # i.e. define self._save_results()
        if save_results:
            ## Dumping Result for detection only now
            assert (
                self._problem_type == OBJECT_DETECTION
            ), "Aborting: save results only works for object detection now."

            self._save_path = setup_save_path(
                old_save_path=self._save_path,
                warn_if_exist=False,
            )

            result_path = os.path.join(self._save_path, "result.txt")

            save_result_df(
                pred=pred,
                data=data,
                detection_classes=self._model.model.CLASSES,
                result_path=result_path,
            )

        if (as_pandas is None and isinstance(data, pd.DataFrame)) or as_pandas is True:
            if self._problem_type == OBJECT_DETECTION:
                pred = save_result_df(
                    pred=pred,
                    data=data,
                    detection_classes=self._model.model.CLASSES,
                    result_path=None,
                )
            else:
                pred = self._as_pandas(data=data, to_be_converted=pred)

        return pred

    @override
    def predict_proba(self, **kwargs):
        pass

    @override
    def evaluate(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        query_data: Optional[list] = None,
        response_data: Optional[list] = None,
        id_mappings: Optional[Union[Dict[str, Dict], Dict[str, pd.Series]]] = None,
        metrics: Optional[Union[str, List[str]]] = None,
        chunk_size: Optional[int] = 1024,
        similarity_type: Optional[str] = "cosine",
        cutoffs: Optional[List[int]] = [1, 5, 10],
        label: Optional[str] = None,
        return_pred: Optional[bool] = False,
        realtime: Optional[bool] = None,
        eval_tool: Optional[str] = None,
        **kwargs,
    ):
        # 1. sanity check for verifying if it is ready for inference
        self._verify_inference_ready()

        # 2. route to matcher if necessary. TODO: refactor this into matching learner.
        # We'll leave it as is for now in the predictor.
        # if self._matcher:
        #     return self._matcher.evaluate(
        #         data=data,
        #         query_data=query_data,
        #         response_data=response_data,
        #         id_mappings=id_mappings,
        #         chunk_size=chunk_size,
        #         similarity_type=similarity_type,
        #         cutoffs=cutoffs,
        #         label=label,
        #         metrics=metrics,
        #         return_pred=return_pred,
        #         realtime=realtime,
        #     )

        # 3. for detection problems. TODO: refactor this into Detection Learner
        # i.e. override self._evaluate method in Detection Learner
        # if self._problem_type == OBJECT_DETECTION:
        #     if realtime:
        #         return NotImplementedError(
        #             f"Current problem type {self._problem_type} does not support realtime predict."
        #         )
        #     if isinstance(data, str):
        #         return evaluate_coco(
        #             predictor=self,
        #             anno_file_or_df=data,
        #             metrics=metrics,
        #             return_pred=return_pred,
        #             eval_tool=eval_tool,
        #         )
        #     else:
        #         data = object_detection_data_to_df(data)
        #         return evaluate_coco(
        #             predictor=self,
        #             anno_file_or_df=data,
        #             metrics=metrics,
        #             return_pred=return_pred,
        #             eval_tool="torchmetrics",
        #         )

        # 4. get return type. TODO: refactor this into sub-learners
        # i.e. define self.get_return_type() -> str
        if self._problem_type == NER:
            ret_type = NER_RET
        else:
            ret_type = LOGITS

        # 5. run predict and get outputs
        outputs = predict(
            predictor=self,
            data=data,
            requires_label=True,
            realtime=realtime,
        )
        logits = extract_from_output(ret_type=ret_type, outputs=outputs)

        # 6. get y_pred and y_true for computing metrics. TODO: refactor this into sub-learners
        # i.e. define self.transform_logits_to_metric_input(logits)
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

        # TODO: refactore this into NER learner
        # i.e. define self.get_y_true_for_metric(data)
        # if self._problem_type == NER:
        #     y_true = self._df_preprocessor.transform_label_for_metric(df=data, tokenizer=self._model.tokenizer)
        # else:
        #     y_true = self._df_preprocessor.transform_label_for_metric(df=data)
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

        # 7. compute metrics. TODO: Refactor this into sub-learners
        # i.e. define self.compute_score_and_get_results()
        results = {}
        # if self._problem_type == NER:
        #     score = compute_score(
        #         metric_data=metric_data,
        #         metric_name=self._eval_metric_name.lower(),
        #     )
        #     score = {k.lower(): v for k, v in score.items()}
        #     if metrics_is_none:
        #         results = score
        #     else:
        #         for per_metric in metrics:
        #             if per_metric.lower() in score:
        #                 results.update({per_metric: score[per_metric.lower()]})
        #             else:
        #                 logger.warning(f"Warning: {per_metric} is not a supported evaluation metric!")
        #         if not results:
        #             results = score  # If the results dict is empty, return all scores.
        # else:
        #     for per_metric in metrics:
        #         score = compute_score(
        #             metric_data=metric_data,
        #             metric_name=per_metric.lower(),
        #         )
        #         results[per_metric] = score
        for per_metric in metrics:
            score = compute_score(
                metric_data=metric_data,
                metric_name=per_metric.lower(),
            )
            results[per_metric] = score

        if return_pred:
            return results, self._as_pandas(data=data, to_be_converted=y_pred_inv)
        else:
            return results

    @override
    def extract_embedding(self, **kwargs):
        pass

    @override
    def fit_summary(self, **kwargs):
        pass

    @override
    def save(self, **kwargs):
        pass

    @override
    def load(self, **kwargs):
        pass

    def _top_k_average(
        self,
        model: nn.Module,
        save_path: str,
        minmax_mode: str,
        is_distill: bool,
        top_k_average_method: Optional[str],
        val_df: pd.DataFrame,
        validation_metric_name: str,
        strategy: Optional[Union[Strategy, str]] = None,
        last_ckpt_path: Optional[str] = None,
        strict_loading: bool = True,
        standalone: bool = True,
        clean_ckpts: bool = True,
    ):
        """Top K average the models at the end of the training

        Parameters
        ----------
        model (nn.Module): model trained
        save_path (str): model save path
        minmax_mode (str): in selecting models, whether a min metric or max metric is preferred
        is_distill (bool): if this is a distillation model
        top_k_average_method (Optional[str]): a string representing top k average method
        val_df (pd.DataFrame): validation data frame for model selection
        validation_metric_name (str): validation metric name
        strategy (Optional[Union[Strategy, str]], optional): Parallelism strategies. Defaults to None.
        last_ckpt_path (Optional[str], optional): _description_. Defaults to None.
        strict_loading (bool, optional): _description_. Defaults to True.
        standalone (bool, optional): _description_. Defaults to True.
        clean_ckpts (bool, optional): _description_. Defaults to True.

        Raises
        ------
        ValueError: raise error when top_k_average_method is not supported
        """
        # FIXME: we need to change validation_metric to evaluation_metric for model choosing
        # since we called self.evaluate. Below is a temporal fix for NER.
        if self._problem_type is not None and self._problem_type == NER:
            validation_metric_name = OVERALL_F1  # seqeval only support overall_f1

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
                        best_score = self.evaluate(val_df, metrics=[validation_metric_name])[validation_metric_name]
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
                            cand_score = self.evaluate(val_df, metrics=[validation_metric_name])[
                                validation_metric_name
                            ]
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

    def _create_lr_callback(self) -> Callback:
        """Creates the learning rate callback
        By default, we use the pl LearningRateMonitor callback
        This can be overriden by child learners to use other learning rate callbacks

        Returns
        --------
        pl.callbacks.LearningRateMonitor: pytorch lightning learning rate callback
        """
        lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
        return lr_callback

    def _create_early_stop_callback(
        self, validation_metric_name: str, minmax_mode: str, task: pl.LightningModule
    ) -> Callback:
        """Creates the callback for early stopping during training
        By default, we use the pl EarlyStopping callback
        This can be overriden by child learners to use other early stopping callbacks

        Parameters
        -----------
        validation_metric_name (str): name of the validation metric
        minmax_mode (str):
            The min/max mode used in selecting model checkpoints.
            - min
                Its means that smaller metric is better.
            - max
                It means that larger metric is better.
        config (DictConfig): OmegaConfig holding the configuration for training
        task (pl.LightningModule): pytorch lightning module for training

        Returns
        --------
        pl.callbacks.EarlyStopping: pytorch lightning early stopping callback
        """
        assert (
            self._config is not None
        ), "self._config is None. You must setup self._config before calling _create_early_stop_callback()"
        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor=task.validation_metric_name,
            patience=self._config.optimization.patience,
            mode=minmax_mode,
            stopping_threshold=get_stopping_threshold(validation_metric_name),
        )

        return early_stopping_callback

    def _create_checkpoint_callback(self, minmax_mode: str, save_path: str, task: pl.LightningModule) -> Callback:
        """Creates the pl checkpoint callback during training
        By default, we use the AutoMMModelCheckpoint, which is a customized checkpoint callback
        This can be overriden by child learners to use other checkpoint callbacks

        Parameters
        -----------
        minmax_mode (str):
            The min/max mode used in selecting model checkpoints.
            - min
                Its means that smaller metric is better.
            - max
                It means that larger metric is better.
        save_path (str): model save path
        self._config (DictConfig): OmegaConfig holding the configuration for training
        task (pl.LightningModule): pytorch lightning module for training

        Returns
        --------
        pl.callbacks.ModelCheckpoint: The pytorch lightning checkpoint callback
        """
        assert (
            self._config is not None
        ), "self._config is None. You must setup self._config before calling _create_checkpoint_callback()"
        checkpoint_callback = AutoMMModelCheckpoint(
            dirpath=save_path,
            save_top_k=self._config.optimization.top_k,
            verbose=True,
            monitor=task.validation_metric_name,
            mode=minmax_mode,
            save_last=True,
        )

        return checkpoint_callback

    def _setup_task_lightning_module(
        self,
        trainable_param_names: List[str],
        mixup_fn: Optional[Mixup],
        loss_func: Optional[nn.Module],
        optimization_kwargs: dict,
        metrics_kwargs: dict,
    ) -> pl.LightningModule:
        """
        Select the correct pl task module for a given problem type.
        By default, we use the default task type a.k.a. LitModule
        This can be overridden by a child learner if needed.


        Parameters
        -----------
        self._config (DictConfig): OmegaConfig dict with all the configs for training
        self._model (nn.Module): pytorch nn.Module to train
        trainable_param_names (List[str]): the list of trainable parameter names
        mixup_fn (Optional[Mixup]): model mixup function that applies different params to each element or whole batch
        loss_func (Optional[nn.Module]): pytorch nn.Module loss function
        self._model_postprocess_fn (Optional[Callable]): postprocess function to apply to the model output (i.e. sigmoid after nn.BCEWithLogitsLoss)
        optimization_kwargs (dict): parameters for optimization
        metrics_kwargs (dict): parameters for metrics

        Returns
        --------
        pl.LightningModule: pytorch lightning module for training task
        """
        assert (
            self._config is not None
        ), "self._config is None. You must setup self._config before calling _setup_task_lightning_module()"
        assert (
            self._model is not None
        ), "self._model is None. You must setup self._model before calling _setup_task_lightning_module()"
        task = LitModule(
            model=self._model,
            loss_func=loss_func,
            efficient_finetune=OmegaConf.select(self._config, "optimization.efficient_finetune"),
            mixup_fn=mixup_fn,
            mixup_off_epoch=OmegaConf.select(self._config, "data.mixup.turn_off_epoch"),
            model_postprocess_fn=self._model_postprocess_fn,
            trainable_param_names=trainable_param_names,
            skip_final_val=OmegaConf.select(self._config, "optimization.skip_final_val", default=False),
            **metrics_kwargs,
            **optimization_kwargs,
        )

        return task

    def _get_metrics_kwargs(
        self, validation_metric_name: str, validation_metric: Optional[Metric], custom_metric_func: Optional[Callable]
    ) -> dict:
        """get the metrics kwargs for the training task

        Parameters
        ----------
        validation_metric_name (str): name of the validation metric
        validation_metric (Optional[Metric]): validation metric object of type torchmetrics.Metric
        custom_metric_func (Optional[Callable]): customized metric function

        Returns
        -------
        dict: dictionary containing the metrics parameters
        """
        metrics_kwargs = dict(
            validation_metric=validation_metric,
            validation_metric_name=validation_metric_name,
            custom_metric_func=custom_metric_func,
        )

        return metrics_kwargs

    def _get_optim_kwargs(self) -> dict:
        """get the optimization kwargs for the training task

        Parameters
        ----------
        self._config (DictConfig): OmegaConf holding the config for training

        Returns
        -------
        dict: dictionary containing the optimization parameters
        """
        assert (
            self._config is not None
        ), "self._config is None. You must setup self._config before calling _get_optim_kwargs()"
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

        return optimization_kwargs

    def _get_train_data_module(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        val_use_training_mode: bool,
    ) -> LightningDataModule:
        """get the data module for training
        By default, we use the default data module a.k.a. BaseDataModule
        This is to be overridden by a child learner if needed.

        Parameters
        ----------
        train_df (pd.DataFrame): training data as in pandas data frame
        val_df (pd.DataFrame): validation data as in pandas data frame
        self._config (DictConfig): OmegaConfig instance holding training configurations
        self._df_preprocessor (MultiModalFeaturePreprocessor): data frame preprocessor to read and preprocess pandas data frames
        sefl._data_processors (List): data processors to read trainig data
        val_use_training_mode (bool): whether to use training mode for validation data

        Returns
        -------
        LightningDataModule: pytorch lightning data module used for loading data during training
        """
        assert (
            self._config is not None
        ), "self._config is None. You must setup self._config before calling _get_train_data_module()"
        assert (
            self._df_preprocessor is not None
        ), "self._df_preprocessor is None. You must setup self._df_preprocessor before calling _get_train_data_module()"
        assert (
            self._data_processors is not None
        ), "self._data_processors is None. You must setup self._data_processors before calling _get_train_data_module()"
        train_dm = BaseDataModule(
            df_preprocessor=self._df_preprocessor,
            data_processors=self._data_processors,
            per_gpu_batch_size=self._config.env.per_gpu_batch_size,
            num_workers=self._config.env.num_workers,
            train_data=train_df,
            validate_data=val_df,
            val_use_training_mode=val_use_training_mode,
        )

        return train_dm

    def _setup_mixup(self) -> Tuple[bool, Optional[Mixup]]:
        """Setup mixup for training

        Parameters
        ----------
        self._config (DictConfig): OmegaConfig instance holding model configurations

        Returns
        -------
        Tuple[bool, Optional[Mixup]]: tuple of mixup_active and mixup_fn (Mixup if mixup is active, None otherwise)
        """
        assert (
            self._config is not None
        ), "self._config must be set before calling _setup_mixup(). Are you calling _setup_mixup() directly? Use .fit() instead."
        mixup_active, mixup_fn = get_mixup(
            model_config=OmegaConf.select(self._config, "model"),
            mixup_config=OmegaConf.select(self._config, "data.mixup"),
            num_classes=self._output_shape,
        )
        if mixup_active and (self._config.env.per_gpu_batch_size == 1 or self._config.env.per_gpu_batch_size % 2 == 1):
            warnings.warn(
                "The mixup is done on the batch."
                "The per_gpu_batch_size should be >1 and even for reasonable operation",
                UserWarning,
            )

        return mixup_active, mixup_fn

    def _setup_validation_metric(
        self, validation_metric_name: Optional[str] = None
    ) -> Tuple[Optional[Metric], Optional[Callable]]:
        """Get validation metric from validation metric name

        Parameters
        ----------
        validation_metric_name (Optional[str]): name for the validation metric

        Returns
        -------
        Tuple[Metric, Union[Callable, None]]: Tuple of validation metric and custom metric function.
        """
        if validation_metric_name is not None:
            validation_metric, custom_metric_func = get_metric(
                metric_name=validation_metric_name,
                num_classes=self._output_shape,
            )
        else:
            validation_metric, custom_metric_func = (None, None)
        return validation_metric, custom_metric_func

    def _get_data_processors(self, advanced_hyperparameters: Optional[Dict]) -> List[object]:
        """Create data processors for training. Reuse the existing data processors if exists

        Parameters
        ----------
        config (DictConfig): OmegaConfig instance holding model configurations
        advanced_hyperparameters (dict): advanced hyperparameters for data processors, such as image transforms, whose values are complex objects
        model (nn.Module): torch model (a.k.a. nn.Module)

        Returns
        TODO: !!! Hot - data processors do not inherit from a common base class!!!!!!
        List[object]: list of required data processors.
        """
        assert (
            self._config is not None
        ), "self._config must be set before calling _get_data_processors. Are you calling _get_data_processors directly? Use .fit() instead"
        assert (
            self._model is not None
        ), "self._model must be set before calling _get_data_processors. Are you calling _get_data_processors directly? Use .fit() instead"
        if self._data_processors is None:
            data_processors = create_fusion_data_processors(
                config=self._config,
                model=self._model,
                advanced_hyperparameters=advanced_hyperparameters,
            )
        else:  # continuing training
            data_processors = self._data_processors
        return data_processors

    def _get_trainable_params(self) -> List[str]:
        """Get trainable parameters for efficient finetuning

        Parameters
        -----------
        self._config (DictConfig): OmegaConfig instance holding model configurations
        self._model (nn.Module): the torch model (a.k.a. nn.Module)

        Returns
        --------
        List[str]: list of trainable parameter names
        """
        assert (
            self._config is not None
        ), "self._config must be set before calling _get_trainable_params. Are you calling _get_trainable_params directly? Use .fit() instead"

        assert (
            self._model is not None
        ), "self._model must be set before calling _get_trainable_params. Are you calling _get_trainable_params directly? Use .fit() instead"
        norm_param_names = get_norm_layer_param_names(self._model)

        trainable_param_names = get_trainable_params_efficient_finetune(
            norm_param_names,
            efficient_finetune=OmegaConf.select(self._config, "optimization.efficient_finetune"),
        )

        return trainable_param_names

    def _create_model(self) -> nn.Module:
        """Creates the model for training based on the config and info in df_preprocessor
        By default we use the create_fusion_model function to create the model
        This can be overriden by a child learner to create custom models

        Parameters
        -----------
        self._config (DictConfig): OmegaConfig instance holding model configurations
        self._df_preprocessor (MultiModalFeaturePreprocessor): data frame preprocessor

        Returns
        --------
        nn.Module: the torch model (a.k.a. nn.Module) to be trained
        """

        assert (
            self._config is not None and self._df_preprocessor is not None
        ), "Config and df_preprocessor must be set before calling _create_model(). Are you calling _create_model() directly? Use .fit() instead."
        ##  this is to be moved into NER learners
        if self._problem_type == NER:
            self._output_shape = len(self._df_preprocessor.label_generator.unique_entity_groups)

        # 5. setup model.
        # create a new model if calling ._fit() for the first time. otherwise re-use the existing self._model
        if self._model is None:
            model = create_fusion_model(
                config=self._config,
                num_classes=self._output_shape,
                classes=self._classes,
                num_numerical_columns=len(self._df_preprocessor.numerical_feature_names),
                num_categories=self._df_preprocessor.categorical_num_categories,
            )
        else:  # continuing training
            model = self._model
        return model

    def _setup_environment(
        self,
        train_df: pd.DataFrame,
        config: dict,
        presets: Optional[str],
        hyperparameters: Optional[Union[str, List[str], Dict]],
        teacher_predictor: Union[str, MultiModalPredictor],
        hpo_mode: bool,
        **hpo_kwargs,
    ) -> Tuple[DictConfig, MultiModalFeaturePreprocessor, int, Optional[Union[Strategy, str]], bool]:
        """Set up training environment parameters: config, df_preprocessor, grad_step, strategy, etc.
        TODO: Can df_preprocessor be moved outside this function?

        Parameters
        -----------
        train_df (pd.DataFrame): training data frame
        presets (Optional[str]): preset to use
        hyperparameters (Optional[Union[str, List[str], Dict]]): user provided hyperparameter overrides
        teacher_predictor (Union[str, MultiModalPredictor]): teacher model for knowledge distillation
        hpo_mode (_type_): if running hpo
        hpo_kwargs (_type_): hpo params

        Returns
        --------
        config: dict
        df_preprocessor: MultiModalFeaturePreprocessor
        grad_step: int
        strategy: Optional[Union[Strategy, str]]
        use_ray_lightning: bool
        """
        config = get_config(
            problem_type=self._problem_type,
            presets=presets,
            config=config,
            overrides=hyperparameters,
            extra=["distiller"] if teacher_predictor is not None else None,
        )

        config = update_config_by_rules(
            problem_type=self._problem_type,
            config=config,
        )

        # 2. set up df_preprocessor if not given yet
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

        # 3. update config given df_preprocessor
        # Avoid passing tabular data with many columns to MultiHeadAttention.
        # If models have additive_attention="auto", we enable it automatically for large tables.

        # NOTE: This part only changes config.model and config.env
        config = update_tabular_config_by_resources(
            config,
            num_numerical_columns=len(df_preprocessor.numerical_feature_names),
            num_categorical_columns=len(df_preprocessor.categorical_num_categories),
        )

        # This part only changes config.model
        config = select_model(config=config, df_preprocessor=df_preprocessor)

        # 22. training environment-dependent configs
        num_gpus = compute_num_gpus(config_num_gpus=config.env.num_gpus, strategy=config.env.strategy)

        precision = infer_precision(num_gpus=num_gpus, precision=config.env.precision)

        grad_steps = compute_grad_steps(config=config, num_gpus=num_gpus)

        use_ray_lightning = "_ray_lightning_plugin" in hpo_kwargs
        num_gpus, strategy = self._compute_strategy(config, hpo_mode, hpo_kwargs, num_gpus, use_ray_lightning)

        config.env.num_gpus = num_gpus
        config.env.precision = precision
        config.env.strategy = strategy if not config.env.strategy == DEEPSPEED_OFFLOADING else DEEPSPEED_OFFLOADING

        self._config = config
        return config, df_preprocessor, grad_steps, strategy, use_ray_lightning

    def _compute_strategy(
        self, config: dict, hpo_mode: bool, hpo_kwargs: dict, num_gpus: int, use_ray_lightning: bool
    ) -> Tuple[int, Optional[Union[Strategy, str]]]:
        """Compute what data parallel strategy to use for training

        Parameters
        -----------
        config (dict): configuration OmegaConfig instance
        hpo_mode (bool): whether to use hpo
        hpo_kwargs (dict): key word arguments for hpo
        num_gpus (int): number of gpus to use
        use_ray_lightning (bool): whether to use ray lightning for hpo

        Returns
        --------
        Tuple[int, Optional[Union[Strategy, str]]]: Tuple containing number of gpus and strategy (num_gpus, strategy)
        """
        if not hpo_mode:
            if num_gpus <= 1:
                if config.env.strategy == DEEPSPEED_OFFLOADING:  # Offloading currently only tested for single GPU
                    assert version.parse(pl.__version__) >= version.parse(
                        DEEPSPEED_MIN_PL_VERSION
                    ), f"For DeepSpeed Offloading to work reliably you need at least pytorch-lightning version {DEEPSPEED_MIN_PL_VERSION}, however, found {pl.__version__}. Please update your pytorch-lightning version."
                    from autogluon.multimodal.optimization.deepspeed import CustomDeepSpeedStrategy

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
        return num_gpus, strategy

    @staticmethod
    def _load_state_dict(
        model: nn.Module,
        state_dict: dict = None,
        path: str = None,
        prefix: str = "model.",
        strict: bool = True,
    ):
        if state_dict is None:
            if os.path.isdir(path + "-dir"):  # deepspeed save checkpoints into a directory
                from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

                convert_zero_checkpoint_to_fp32_state_dict(path + "-dir", path)
                shutil.rmtree(path + "-dir")
                state_dict = torch.load(path, map_location=torch.device("cpu"))["state_dict"]
            else:
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

    def _verify_inference_ready(self):
        if not self._fit_called:
            if self._problem_type and not self.problem_property.support_zero_shot:
                raise RuntimeError(
                    f"problem_type='{self._problem_type}' does not support running inference directly. "
                    f"You need to call `predictor.fit()`, or load a predictor first before "
                    f"running `predictor.predict()`, `predictor.evaluate()` or `predictor.extract_embedding()`."
                )

    # TODO: consider renaming this to something like prepare predict?
    def _on_predict_start(
        self,
        data: Union[pd.DataFrame, dict, list],
        requires_label: bool,
    ):
        if self._column_types is None:
            data = data_to_df(data=data)
            allowable_dtypes, fallback_dtype = infer_dtypes_by_model_names(model_config=self._config.model)
            column_types = infer_column_types(
                data=data,
                allowable_column_types=allowable_dtypes,
                fallback_column_type=fallback_dtype,
            )
            if self._label_column and self._label_column in data.columns:
                column_types = infer_label_column_type_by_problem_type(
                    column_types=column_types,
                    label_columns=self._label_column,
                    problem_type=self._problem_type,
                    data=data,
                )

            if self._problem_type == OBJECT_DETECTION:
                column_types = infer_rois_column_type(
                    column_types=column_types,
                    data=data,
                )
        else:  # called .fit() or .load()
            column_names = list(self._column_types.keys())
            # remove label column since it's not required in inference.
            column_names.remove(self._label_column)
            data = data_to_df(
                data=data,
                required_columns=self._df_preprocessor.required_feature_names,
                all_columns=column_names,
            )
            column_types = self._column_types
            column_types_copy = copy.deepcopy(column_types)
            for col_name, col_type in column_types.items():
                if col_type in [IMAGE_BYTEARRAY, IMAGE_PATH]:
                    if is_image_column(data=data[col_name], col_name=col_name, image_type=IMAGE_PATH):
                        image_type = IMAGE_PATH
                    elif is_image_column(
                        data=data[col_name],
                        col_name=col_name,
                        image_type=IMAGE_BYTEARRAY,
                    ):
                        image_type = IMAGE_BYTEARRAY
                    else:
                        image_type = col_type
                    if col_type != image_type:
                        column_types_copy[col_name] = image_type
            self._df_preprocessor._column_types = column_types_copy

        if self._df_preprocessor is None:
            df_preprocessor = init_df_preprocessor(
                config=self._config,
                column_types=column_types,
                label_column=self._label_column,
                train_df_x=data,  # TODO: drop label like in line 884?
                train_df_y=data[self._label_column] if self._label_column else None,
            )
        else:  # called .fit() or .load()
            df_preprocessor = self._df_preprocessor

        data_processors = copy.copy(self._data_processors)
        # For prediction data with no labels provided.
        if not requires_label:
            data_processors.pop(LABEL, None)

        return data, df_preprocessor, data_processors

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
            from autogluon.multimodal.optimization.deepspeed import CustomDeepSpeedStrategy

            strategy = CustomDeepSpeedStrategy(
                stage=3,
                offload_optimizer=True,
                offload_parameters=False,
                allgather_bucket_size=self._config.env.deepspeed_allgather_size,
                reduce_bucket_size=self._config.env.deepspeed_allreduce_size,
            )
            norm_param_names = get_norm_layer_param_names(self._model)
            trainable_param_names = get_trainable_params_efficient_finetune(
                norm_param_names,
                efficient_finetune=OmegaConf.select(self._config, "optimization.efficient_finetune"),
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

        callbacks = []

        if strategy == "ddp":
            if self._problem_type != OBJECT_DETECTION:
                raise NotImplementedError(f"inference using ddp is only implemented for {OBJECT_DETECTION}")
            else:
                pred_writer = DDPCacheWriter(pipeline=self._problem_type, write_interval="epoch")
                callbacks = [pred_writer]

        # TODO: refactor into sub-learners
        # if self._problem_type == NER:
        #     task = NerLitModule(
        #         model=self._model,
        #         model_postprocess_fn=self._model_postprocess_fn,
        #         efficient_finetune=OmegaConf.select(self._config, "optimization.efficient_finetune"),
        #         trainable_param_names=trainable_param_names,
        #         **optimization_kwargs,
        #     )
        # elif self._problem_type == OBJECT_DETECTION:
        #     task = MMDetLitModule(
        #         model=self._model,
        #         **optimization_kwargs,
        #     )
        # else:
        #     task = LitModule(
        #         model=self._model,
        #         model_postprocess_fn=self._model_postprocess_fn,
        #         efficient_finetune=OmegaConf.select(self._config, "optimization.efficient_finetune"),
        #         trainable_param_names=trainable_param_names,
        #         **optimization_kwargs,
        #     )

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
                devices=get_available_devices(num_gpus=num_gpus, auto_select_gpus=self._config.env.auto_select_gpus),
                num_nodes=self._config.env.num_nodes,
                precision=precision,
                strategy=strategy,
                benchmark=False,
                enable_progress_bar=self._enable_progress_bar,
                deterministic=self._config.env.deterministic,
                max_epochs=-1,  # Add max_epochs to disable warning
                logger=False,
                callbacks=callbacks,
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
                    return_predictions=not callbacks,
                )

                if strategy == "ddp":
                    if evaluator.global_rank != 0:
                        sys.exit(f"Prediction finished, exit the process with global_rank={evaluator.global_rank}...")
                    else:
                        outputs = pred_writer.collect_all_gpu_results(num_gpus=num_gpus)
                elif self._problem_type == OBJECT_DETECTION:
                    # reformat single gpu output for object detection
                    # outputs shape: num_batch, 1(["bbox"]), batch_size, 80, n, 5
                    # output LABEL if exists for evaluations
                    outputs = [
                        {BBOX: bbox, LABEL: ele[LABEL][i]} if LABEL in ele else {BBOX: bbox}
                        for ele in outputs
                        for i, bbox in enumerate(ele[BBOX])
                    ]

        return outputs
