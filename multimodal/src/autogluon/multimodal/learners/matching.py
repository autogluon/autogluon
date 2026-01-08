from __future__ import annotations

import copy
import json
import logging
import operator
import os
import pickle
import time
import warnings
from datetime import timedelta
from typing import Callable, Dict, List, Optional, Union

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from torch import nn

from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.common.utils.resource_utils import ResourceManager

from .. import version as ag_version
from ..constants import (
    BEST,
    BEST_K_MODELS_FILE,
    BINARY,
    FEATURES,
    GREEDY_SOUP,
    IMAGE_TEXT_SIMILARITY,
    LABEL,
    LAST_CHECKPOINT,
    MAX,
    MIN,
    MODEL_CHECKPOINT,
    MULTICLASS,
    PAIR,
    PROBABILITY,
    QUERY,
    RESPONSE,
    TEXT,
    UNIFORM_SOUP,
    Y_PRED,
    Y_PRED_PROB,
    Y_TRUE,
)
from ..data import (
    BaseDataModule,
    MultiModalFeaturePreprocessor,
    create_fusion_data_processors,
    data_to_df,
    infer_column_types,
    infer_dtypes_by_model_names,
    init_df_preprocessor,
)
from ..models import is_lazy_weight_tensor, select_model
from ..optim import (
    MatcherLitModule,
    compute_ranking_score,
    compute_score,
    get_matcher_loss_func,
    get_matcher_miner_func,
    get_torchmetric,
)
from ..utils import (
    average_checkpoints,
    compute_semantic_similarity,
    convert_data_for_ranking,
    create_siamese_model,
    customize_model_names,
    extract_from_output,
    get_config,
    get_dir_ckpt_paths,
    get_load_ckpt_paths,
    get_local_pretrained_config_paths,
    hyperparameter_tune,
    matcher_presets,
    on_fit_end_message,
    save_pretrained_model_configs,
    split_hyperparameters,
    update_config_by_rules,
)
from ..utils.problem_types import PROBLEM_TYPES_REG
from .base import BaseLearner

pl_logger = logging.getLogger("lightning")
pl_logger.propagate = False  # https://github.com/Lightning-AI/lightning/issues/4621
logger = logging.getLogger(__name__)


class MatchingLearner(BaseLearner):
    """
    MatchingLearner is a framework to learn/extract embeddings for multimodal data including image, text, and tabular.
    These embeddings can be used e.g. with cosine-similarity to find items with similar semantic meanings.
    This can be useful for computing the semantic similarity of two items, semantic search, paraphrase mining, etc.
    """

    def __init__(
        self,
        query: Optional[Union[str, List[str]]] = None,
        response: Optional[Union[str, List[str]]] = None,
        label: Optional[str] = None,
        match_label: Optional[Union[int, str]] = None,
        problem_type: Optional[str] = None,
        presets: Optional[str] = None,
        eval_metric: Optional[str] = None,
        hyperparameters: Optional[dict] = None,
        path: Optional[str] = None,
        verbosity: Optional[int] = 3,
        warn_if_exist: Optional[bool] = True,
        enable_progress_bar: Optional[bool] = None,
        pretrained: Optional[bool] = True,
        validation_metric: Optional[str] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        query
            Column names of query data.
        response
            Column names of response data. If no label column is provided,
            query and response columns form positive pairs.
        label
            Name of the label column.
        match_label
            The label class that indicates the <query, response> pair is counted as "match".
            This is used when the problem_type is one of the matching problem types, and when the labels are binary.
            For example, the label column can contain ["match", "not match"]. And match_label can be "match".
            It is similar as the "pos_label" in F1-score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
            Internally, we will set match_label to self.class_labels[1] by default.
        problem_type
            Type of matching problem if the label column is available.
            This could be binary, multiclass, or regression
            if the label column contains binary, multiclass, or numeric labels.
            If `problem_type = None`, the prediction problem type is inferred
            based on the label-values in provided dataset.
        presets
            Presets regarding model quality, e.g., best_quality, high_quality, and medium_quality.
        eval_metric
            Evaluation metric name. If `eval_metric = None`, it is automatically chosen based on `problem_type`.
            Defaults to 'roc_auc' for binary classification and 'spearmanr' for multiclass classification and regression.
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
            Defaults to 'roc_auc' for binary classification and 'spearmanr' for multiclass classification and regression.
        """
        if eval_metric is not None and not isinstance(eval_metric, str):
            eval_metric = eval_metric.name

        if isinstance(query, str):
            query = [query]
        if query:
            assert all(isinstance(q, str) for q in query)

        if isinstance(response, str):
            response = [response]
        if response:
            assert all(isinstance(r, str) for r in response)

        self._query = query
        self._response = response
        self._data_format = PAIR  # TODO: Support Triplet
        self._match_label = match_label
        self._label_column = label
        self._problem_type = None  # always infer problem type for matching.
        self._pipeline = problem_type.lower() if problem_type is not None else None
        self._presets = presets.lower() if presets else None
        self._eval_metric_name = eval_metric.lower() if eval_metric else None
        self._eval_metric_func = None
        self._validation_metric_name = validation_metric.lower() if validation_metric else None
        self._minmax_mode = None
        self._hyperparameters = hyperparameters
        self._advanced_hyperparameters = None
        self._hyperparameter_tune_kwargs = None
        self._output_shape = None
        self._save_path = path
        self._ckpt_path = None
        self._pretrained_path = None
        self._pretrained = pretrained
        self._config = None
        self._query_config = None
        self._response_config = None
        self._query_df_preprocessor = None
        self._response_df_preprocessor = None
        self._label_df_preprocessor = None
        self._column_types = None
        self._query_processors = None
        self._response_processors = None
        self._label_processors = None
        self._query_model = None
        self._response_model = None
        self._is_hpo = False
        self._resume = False
        self._fit_called = False
        self._train_data = None
        self._tuning_data = None
        self._verbosity = verbosity
        self._warn_if_exist = warn_if_exist
        self._enable_progress_bar = enable_progress_bar if enable_progress_bar is not None else True
        self._fit_args = None
        self._total_train_time = None
        self._best_score = None

        self._log_filters = [
            ".*does not have many workers.* in the `DataLoader` init to improve performance.*",
            "Checkpoint directory .* exists and is not empty.",
        ]

    @property
    def query(self):
        return self._query

    @property
    def response(self):
        return self._response

    @property
    def match_label(self):
        return self._match_label

    @property
    def path(self):
        return self._save_path

    @property
    def label(self):
        return self._label_column

    @property
    def problem_type(self):
        if self._pipeline and self._problem_type:
            return f"{self._pipeline}_{self._problem_type}"
        elif self._pipeline:
            return self._pipeline
        else:
            return self._problem_type

    @property
    def problem_property(self):
        return PROBLEM_TYPES_REG.get(self._pipeline)

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
        return sum(p.numel() if not is_lazy_weight_tensor(p) else 0 for p in self._query_model.parameters())

    @property
    def trainable_parameters(self) -> int:
        return sum(
            p.numel() if not is_lazy_weight_tensor(p) else 0 for p in self._query_model.parameters() if p.requires_grad
        )

    @property
    def model_size(self) -> float:
        """
        Returns the model size in Megabyte.
        """
        model_size = sum(
            p.numel() * p.element_size() if not is_lazy_weight_tensor(p) else 0 for p in self._query_model.parameters()
        )
        return model_size * 1e-6  # convert to megabytes

    # This func is required by the abstract trainer of TabularPredictor.
    def set_verbosity(self, verbosity: int):
        """
        Set the verbosity level of the log.

        Parameters
        ----------
        verbosity
            The verbosity level

        """
        self._verbosity = verbosity
        set_logger_verbosity(verbosity, logger=logger)

    def _init_pretrained(self):
        if self._config is None:
            # split out the hyperparameters whose values are complex objects
            hyperparameters, advanced_hyperparameters = split_hyperparameters(self._hyperparameters)
            self._config = get_config(
                problem_type=self._pipeline,
                presets=self._presets,
                overrides=hyperparameters,
                extra=["matcher"],
            )
        else:
            advanced_hyperparameters = None

        assert len(self._config.model.names) == 1, (
            f"Zero shot mode only supports using one model, but detects multiple models {self._config.model.names}"
        )

        if self._query_config is None:
            self._query_config = copy.deepcopy(self._config)
            # customize config model names to make them consistent with model prefixes.
            self._query_config.model, query_advanced_hyperparameters = customize_model_names(
                config=self._query_config.model,
                customized_names=[f"{n}_{QUERY}" for n in self._query_config.model.names],
                advanced_hyperparameters=advanced_hyperparameters,
            )
        else:
            query_advanced_hyperparameters = (None,)

        if self._response_config is None:
            self._response_config = copy.deepcopy(self._config)
            # customize config model names to make them consistent with model prefixes.
            self._response_config.model, response_advanced_hyperparameters = customize_model_names(
                config=self._response_config.model,
                customized_names=[f"{n}_{RESPONSE}" for n in self._response_config.model.names],
                advanced_hyperparameters=advanced_hyperparameters,
            )
        else:
            response_advanced_hyperparameters = None

        if self._query_model is None or self._response_model is None:
            self._query_model, self._response_model = create_siamese_model(
                query_config=self._query_config,
                response_config=self._response_config,
                pretrained=self._pretrained,
            )

        self._query_processors, self._response_processors, self._label_processors = self._get_matcher_data_processors(
            query_model=self._query_model,
            query_config=self._query_config,
            response_model=self._response_model,
            response_config=self._response_config,
            query_advanced_hyperparameters=query_advanced_hyperparameters,
            response_advanced_hyperparameters=response_advanced_hyperparameters,
        )

    def _ensure_inference_ready(self):
        if not self._fit_called:
            self._init_pretrained()

    def prepare_fit_args(
        self,
        time_limit: int,
        seed: int,
        id_mappings: Dict,
        standalone: Optional[bool] = True,
        clean_ckpts: Optional[bool] = True,
    ):
        if time_limit is not None:
            time_limit = timedelta(seconds=time_limit)
        self._fit_args = dict(
            id_mappings=id_mappings,
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
            )  # TODO: add more continuous training arguments
            self._fit_args.update(continuous_train_args)

    def update_attributes(
        self,
        config: Dict,
        query_config: DictConfig,
        response_config: DictConfig,
        query_model: nn.Module,
        response_model: nn.Module,
        query_df_preprocessor: MultiModalFeaturePreprocessor,
        response_df_preprocessor: MultiModalFeaturePreprocessor,
        label_df_preprocessor: MultiModalFeaturePreprocessor,
        query_processors: Dict,
        response_processors: Dict,
        label_processors: Dict,
        best_score: Optional[float] = None,
    ):
        self._config = config
        self._query_config = query_config
        self._response_config = response_config
        self._query_model = query_model
        self._response_model = response_model
        self._query_df_preprocessor = query_df_preprocessor
        self._response_df_preprocessor = response_df_preprocessor
        self._label_df_preprocessor = label_df_preprocessor
        self._query_processors = query_processors
        self._response_processors = response_processors
        self._label_processors = label_processors
        if best_score:
            self._best_score = best_score

    def execute_fit(self):
        if self._is_hpo:
            self._fit_args["learner"] = self
            hyperparameter_tune(
                hyperparameter_tune_kwargs=self._hyperparameter_tune_kwargs,
                resources=dict(num_gpus=ResourceManager.get_gpu_count_torch()),  # TODO: allow customizing GPUs
                is_matching=True,
                **self._fit_args,
            )
            return dict()
        else:
            attributes = self.fit_per_run(**self._fit_args)
            self.update_attributes(**attributes)  # only update attributes for non-HPO mode
            return attributes

    def on_fit_end(
        self,
        training_start: float,
        standalone: Optional[bool] = True,
        clean_ckpts: Optional[bool] = True,
    ):
        self._fit_called = True
        if not self._is_hpo:
            # top_k_average is called inside hyperparameter_tune() when building the final predictor.
            self.top_k_average(
                save_path=self._save_path,
                top_k_average_method=self._config.optim.top_k_average_method,
                standalone=standalone,
                clean_ckpts=clean_ckpts,
            )

        training_end = time.time()
        self._total_train_time = training_end - training_start
        # TODO(?) We should have a separate "_post_training_event()" for logging messages.
        logger.info(on_fit_end_message(self._save_path))

    def fit(
        self,
        train_data: pd.DataFrame,
        id_mappings: Optional[Union[Dict[str, Dict], Dict[str, pd.Series]]] = None,
        presets: Optional[str] = None,
        tuning_data: Optional[pd.DataFrame] = None,
        time_limit: Optional[int] = None,
        save_path: Optional[str] = None,
        hyperparameters: Optional[Union[str, Dict, List[str]]] = None,
        column_types: Optional[dict] = None,
        holdout_frac: Optional[float] = None,
        hyperparameter_tune_kwargs: Optional[dict] = None,
        seed: Optional[int] = 123,
        standalone: Optional[bool] = True,
        clean_ckpts: Optional[bool] = True,
        **kwargs,
    ):
        """
        Fit MatchingLearner. Train the model to learn embeddings to simultaneously maximize and minimize
        the semantic similarities of positive and negative pairs.
        The data may contain image, text, numeric, or categorical features.

        Parameters
        ----------
        train_data
            A dataframe, containing the query data, response data, and their relevance scores. For example,
            | query_col1  | query_col2 | response_col1 | response_col2 | relevance_score |
            |-------------|------------|---------------|---------------|-----------------|
            | ....        | ....       | ....          | ...           | ...             |
            | ....        | ....       | ....          | ...           | ...             |
        id_mappings
             Id-to-content mappings. The contents can be text, image, etc.
             This is used when the dataframe contains the query/response identifiers instead of their contents.
        presets
            Presets regarding model quality, e.g., best_quality, high_quality, and medium_quality.
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
        seed
            The random seed to use for this training run.

        Returns
        -------
        An "MatchingLearner" object (itself).
        """
        self.setup_save_path(save_path=save_path)
        training_start = self.on_fit_start(presets=presets)
        self.infer_problem_type(train_data=train_data)
        self.prepare_train_tuning_data(
            train_data=train_data,
            tuning_data=tuning_data,
            holdout_frac=holdout_frac,
            seed=seed,
        )
        self.infer_column_types(column_types=column_types)
        self.infer_output_shape()
        self.infer_validation_metric(is_matching=True)
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
            id_mappings=id_mappings,
        )
        self.execute_fit()
        self.on_fit_end(
            training_start=training_start,
            standalone=standalone,
            clean_ckpts=clean_ckpts,
        )
        return self

    def _get_matcher_df_preprocessor(
        self,
        data: pd.DataFrame,
        column_types: Dict,
        query_config: Optional[DictConfig] = None,
        response_config: Optional[DictConfig] = None,
        query_columns: Optional[List] = None,
        response_columns: Optional[List] = None,
    ):
        if query_columns is None:
            query_df_preprocessor = None
        elif self._query_df_preprocessor is None and all(v is not None for v in [query_columns, query_config]):
            query_df_preprocessor = init_df_preprocessor(
                config=query_config,
                column_types={k: column_types[k] for k in query_columns},
                train_df_x=data[query_columns],
            )
        else:  # continuing training
            query_df_preprocessor = self._query_df_preprocessor

        if response_columns is None:
            response_df_preprocessor = None
        elif self._response_df_preprocessor is None and all(
            v is not None for v in [response_columns, response_config]
        ):
            response_df_preprocessor = init_df_preprocessor(
                config=response_config,
                column_types={k: column_types[k] for k in response_columns},
                train_df_x=data[response_columns],
            )
        else:  # continuing training
            response_df_preprocessor = self._response_df_preprocessor

        if self._label_column is None:
            label_df_preprocessor = None
        elif (
            self._label_df_preprocessor is None and response_config is not None and self._label_column in column_types
        ):
            label_df_preprocessor = init_df_preprocessor(
                config=response_config,
                column_types={self._label_column: column_types[self._label_column]},
                label_column=self._label_column,
                train_df_y=data[self._label_column],
            )
        else:  # continuing training
            label_df_preprocessor = self._label_df_preprocessor

        return query_df_preprocessor, response_df_preprocessor, label_df_preprocessor

    def _get_matcher_data_processors(
        self,
        query_model: Optional[nn.Module] = None,
        query_config: Optional[DictConfig] = None,
        response_model: Optional[nn.Module] = None,
        response_config: Optional[DictConfig] = None,
        query_advanced_hyperparameters: Optional[Dict] = None,
        response_advanced_hyperparameters: Optional[Dict] = None,
    ):
        if query_model is None:
            query_processors = None
        elif self._query_processors is None and all(v is not None for v in [query_model, query_config]):
            query_processors = create_fusion_data_processors(
                model=query_model,
                config=query_config,
                requires_label=False,
                requires_data=True,
                advanced_hyperparameters=query_advanced_hyperparameters,
            )
        else:  # continuing training
            query_processors = self._query_processors

        if response_model is None:
            response_processors = None
        elif self._response_processors is None and all(v is not None for v in [response_model, response_config]):
            response_processors = create_fusion_data_processors(
                model=response_model,
                config=response_config,
                requires_label=False,
                requires_data=True,
                advanced_hyperparameters=response_advanced_hyperparameters,
            )
        else:  # continuing training
            response_processors = self._response_processors

        # only need labels for the response model
        if response_model is None:
            label_processors = None
        elif self._label_processors is None and all(
            v is not None for v in [self._label_column, response_model, response_config]
        ):
            label_processors = create_fusion_data_processors(
                model=response_model,
                config=response_config,
                requires_label=True,
                requires_data=False,
            )
        else:  # continuing training
            label_processors = self._label_processors

        return query_processors, response_processors, label_processors

    def get_config_per_run(self, config, hyperparameters):
        config = get_config(
            problem_type=self._pipeline,
            presets=self._presets,
            config=config,
            overrides=hyperparameters,  # don't use self._hyperparameters due to HPO.
            extra=["matcher"],
        )
        config = update_config_by_rules(
            problem_type=self._pipeline,
            config=config,
        )
        config = self.update_strategy_by_env(config=config)
        return config

    def on_fit_per_run_end(
        self,
        trainer: pl.Trainer,
        config: DictConfig,
        query_config: DictConfig,
        response_config: DictConfig,
        query_model: nn.Module,
        response_model: nn.Module,
        query_df_preprocessor: MultiModalFeaturePreprocessor,
        response_df_preprocessor: MultiModalFeaturePreprocessor,
        label_df_preprocessor: MultiModalFeaturePreprocessor,
        query_processors: Dict,
        response_processors: Dict,
        label_processors: Dict,
        save_path: str,
        standalone: bool,
    ):
        self.clean_trainer_processes(trainer=trainer, is_train=True)
        self.save(
            path=save_path,
            standalone=standalone,
            config=config,
            query_config=query_config,
            response_config=response_config,
            query_model=query_model,
            response_model=response_model,
            query_df_preprocessor=query_df_preprocessor,
            response_df_preprocessor=response_df_preprocessor,
            label_df_preprocessor=label_df_preprocessor,
            query_processors=query_processors,
            response_processors=response_processors,
            label_processors=label_processors,
            fit_called=True,  # fit is called on one run.
            save_model=False,  # The final model will be saved in top_k_average
        )

    def fit_per_run(
        self,
        id_mappings: Union[Dict[str, Dict], Dict[str, pd.Series]],
        max_time: timedelta,
        save_path: str,
        ckpt_path: str,
        resume: bool,
        enable_progress_bar: bool,
        seed: int,
        config: Optional[str] = None,
        hyperparameters: Optional[Union[str, Dict, List[str]]] = None,
        advanced_hyperparameters: Optional[Dict] = None,
        standalone: bool = True,
        clean_ckpts: bool = True,
    ):
        self.on_fit_per_run_start(seed=seed, save_path=save_path)
        config = self.get_config_per_run(config=config, hyperparameters=hyperparameters)

        if self._query_config is None:
            query_config = copy.deepcopy(config)
            # customize config model names to make them consistent with model prefixes.
            query_config.model, query_advanced_hyperparameters = customize_model_names(
                config=query_config.model,
                customized_names=[f"{n}_{QUERY}" for n in query_config.model.names],
                advanced_hyperparameters=advanced_hyperparameters,
            )
        else:
            query_config = self._query_config
            query_advanced_hyperparameters = None

        if self._response_config is None:
            response_config = copy.deepcopy(config)
            # customize config model names to make them consistent with model prefixes.
            response_config.model, response_advanced_hyperparameters = customize_model_names(
                config=response_config.model,
                customized_names=[f"{n}_{RESPONSE}" for n in response_config.model.names],
                advanced_hyperparameters=advanced_hyperparameters,
            )
        else:
            response_config = self._response_config
            response_advanced_hyperparameters = None

        query_df_preprocessor, response_df_preprocessor, label_df_preprocessor = self._get_matcher_df_preprocessor(
            data=self._train_data,
            column_types=self._column_types,
            query_config=query_config,
            response_config=response_config,
            query_columns=self._query,
            response_columns=self._response,
        )

        query_config = select_model(config=query_config, df_preprocessor=query_df_preprocessor, strict=False)
        response_config = select_model(config=response_config, df_preprocessor=response_df_preprocessor, strict=False)

        if self._query_model is None or self._response_model is None:
            query_model, response_model = create_siamese_model(
                query_config=query_config,
                response_config=response_config,
                pretrained=self._pretrained,
            )
        else:  # continuing training
            query_model = self._query_model
            response_model = self._response_model

        query_processors, response_processors, label_processors = self._get_matcher_data_processors(
            query_model=query_model,
            query_config=query_config,
            response_model=response_model,
            response_config=response_config,
            query_advanced_hyperparameters=query_advanced_hyperparameters,
            response_advanced_hyperparameters=response_advanced_hyperparameters,
        )

        query_processors_count = {k: len(v) for k, v in query_processors.items()}
        logger.debug(f"query_processors_count: {query_processors_count}")
        response_processors_count = {k: len(v) for k, v in response_processors.items()}
        logger.debug(f"response_processors_count: {response_processors_count}")
        if label_processors:
            label_processors_count = {k: len(v) for k, v in label_processors.items()}
            logger.debug(f"label_processors_count: {label_processors_count}")

        validation_metric, custom_metric_func = get_torchmetric(
            metric_name=self._validation_metric_name,
            num_classes=self._output_shape,
            is_matching=self._pipeline in matcher_presets.list_keys(),
            problem_type=self._problem_type,
        )
        logger.debug(f"validation_metric_name: {self._validation_metric_name}")
        logger.debug(f"validation_metric: {validation_metric}")
        logger.debug(f"custom_metric_func: {custom_metric_func}")

        loss_func = get_matcher_loss_func(
            data_format=self._data_format,
            problem_type=self._problem_type,
            loss_type=config.matcher.loss.type,
            pos_margin=config.matcher.loss.pos_margin,
            neg_margin=config.matcher.loss.neg_margin,
            distance_type=config.matcher.distance.type,
        )

        miner_func = None
        if self._problem_type == BINARY:
            miner_func = get_matcher_miner_func(
                miner_type=config.matcher.miner.type,
                pos_margin=config.matcher.miner.pos_margin,
                neg_margin=config.matcher.miner.neg_margin,
                distance_type=config.matcher.distance.type,
            )

        if max_time == timedelta(seconds=0):
            return dict(
                config=config,
                query_config=query_config,
                response_config=response_config,
                query_model=query_model,
                response_model=response_model,
                query_df_preprocessor=query_df_preprocessor,
                response_df_preprocessor=response_df_preprocessor,
                label_df_preprocessor=label_df_preprocessor,
                query_processors=query_processors,
                response_processors=response_processors,
                label_processors=label_processors,
            )

        df_preprocessors = [query_df_preprocessor, response_df_preprocessor, label_df_preprocessor]
        data_processors = [query_processors, response_processors, label_processors]
        df_preprocessors = [item for item in df_preprocessors if item is not None]
        data_processors = [item for item in data_processors if item is not None]
        assert len(df_preprocessors) == len(data_processors)

        datamodule = BaseDataModule(
            df_preprocessor=df_preprocessors,
            data_processors=data_processors,
            per_gpu_batch_size=config.env.per_gpu_batch_size,
            num_workers=config.env.num_workers,
            train_data=self._train_data,
            validate_data=self._tuning_data,
            id_mappings=id_mappings,
        )
        optim_kwargs = dict(
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
        )
        metrics_kwargs = dict(
            validation_metric=validation_metric,
            validation_metric_name=self._validation_metric_name,
            custom_metric_func=custom_metric_func,
        )

        if self._match_label is not None:
            match_label = label_df_preprocessor.label_generator.transform([self._match_label]).item()
        else:
            match_label = None

        litmodule = MatcherLitModule(
            query_model=query_model,
            response_model=response_model,
            match_label=match_label,
            loss_func=loss_func,
            miner_func=miner_func,
            **metrics_kwargs,
            **optim_kwargs,
        )
        callbacks = self.get_callbacks_per_run(save_path=save_path, config=config, litmodule=litmodule)
        tb_logger = self.get_tb_logger(save_path=save_path)
        num_gpus, strategy = self.get_num_gpus_and_strategy_per_run(config=config)
        precision = self.get_precision_per_run(num_gpus=num_gpus, precision=config.env.precision)
        grad_steps = self.get_grad_steps(num_gpus=num_gpus, config=config)
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
            trainer=trainer,
            standalone=standalone,
            save_path=save_path,
            config=config,
            query_config=query_config,
            response_config=response_config,
            query_model=query_model,
            response_model=response_model,
            query_df_preprocessor=query_df_preprocessor,
            response_df_preprocessor=response_df_preprocessor,
            label_df_preprocessor=label_df_preprocessor,
            query_processors=query_processors,
            response_processors=response_processors,
            label_processors=label_processors,
        )

        return dict(
            config=config,
            query_config=query_config,
            response_config=response_config,
            query_model=query_model,
            response_model=response_model,
            query_df_preprocessor=query_df_preprocessor,
            response_df_preprocessor=response_df_preprocessor,
            label_df_preprocessor=label_df_preprocessor,
            query_processors=query_processors,
            response_processors=response_processors,
            label_processors=label_processors,
            best_score=trainer.callback_metrics[f"val_{self._validation_metric_name}"].item(),
        )

    def top_k_average(
        self,
        save_path: str,
        top_k_average_method: str,
        last_ckpt_path: Optional[str] = None,
        standalone: Optional[bool] = True,
        clean_ckpts: Optional[bool] = True,
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
                        reverse=(self._minmax_mode == MAX),
                    )
                ]
                if top_k_average_method == GREEDY_SOUP:
                    # Select the ingredients based on the methods proposed in paper
                    #  "Model soups: averaging weights of multiple fine-tuned models improves accuracy without
                    #  increasing inference time", https://arxiv.org/pdf/2203.05482.pdf
                    monitor_op = {MIN: operator.le, MAX: operator.ge}[self._minmax_mode]

                    logger.info(f"Start to fuse {len(top_k_model_paths)} checkpoints via the greedy soup algorithm.")

                    ingredients = [top_k_model_paths[0]]
                    self._load_state_dict(path=top_k_model_paths[0])

                    if self._pipeline == IMAGE_TEXT_SIMILARITY:
                        best_score = self._evaluate_symmetric_ranking(self._tuning_data)
                    else:
                        best_score = self.evaluate(self._tuning_data, metrics=[self._validation_metric_name])[
                            self._validation_metric_name
                        ]
                    for i in range(1, len(top_k_model_paths)):
                        cand_avg_state_dict = average_checkpoints(
                            checkpoint_paths=ingredients + [top_k_model_paths[i]],
                        )
                        self._load_state_dict(state_dict=cand_avg_state_dict)
                        if self._pipeline == IMAGE_TEXT_SIMILARITY:
                            cand_score = self._evaluate_symmetric_ranking(self._tuning_data)
                        else:
                            cand_score = self.evaluate(self._tuning_data, metrics=[self._validation_metric_name])[
                                self._validation_metric_name
                            ]
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
        self._load_state_dict(state_dict=avg_state_dict)

        self._query_model, self._response_model = create_siamese_model(
            query_config=self._query_config,
            response_config=self._response_config,
            query_model=self._query_model,
            response_model=self._response_model,
        )

        task = MatcherLitModule(
            query_model=self._query_model,
            response_model=self._response_model,
        )

        checkpoint = {"state_dict": task.state_dict()}
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

    def _on_predict_start(
        self,
        data: Union[pd.DataFrame, dict, list],
        id_mappings: Union[Dict[str, Dict], Dict[str, pd.Series]],
        requires_label: bool,
        signature: Optional[str] = None,
    ):
        assert signature in [QUERY, RESPONSE, None]

        data = data_to_df(data=data, header=signature)

        if self._column_types is None:
            if signature is None or signature == QUERY:
                allowable_query_dtypes, fallback_query_dtype = infer_dtypes_by_model_names(
                    model_config=self._query_config.model
                )
            if signature is None or signature == RESPONSE:
                allowable_response_dtypes, fallback_response_dtype = infer_dtypes_by_model_names(
                    model_config=self._response_config.model
                )
            if signature == QUERY:
                allowable_dtypes = allowable_query_dtypes
                fallback_dtype = fallback_query_dtype
            elif signature == RESPONSE:
                allowable_dtypes = allowable_response_dtypes
                fallback_dtype = fallback_response_dtype
            else:
                # TODO: consider that query and response have different modalities.
                assert sorted(allowable_query_dtypes) == sorted(allowable_response_dtypes)
                assert fallback_query_dtype == fallback_response_dtype
                allowable_dtypes = allowable_query_dtypes
                fallback_dtype = fallback_query_dtype

            column_types = infer_column_types(
                data=data,
                label_columns=self._label_column,
                problem_type=self._problem_type,
                allowable_column_types=allowable_dtypes,
                fallback_column_type=fallback_dtype,
                id_mappings=id_mappings,
            )
        else:  # called .fit() or .load()
            column_types = self._column_types

        query_config = None
        query_model = None
        response_config = None
        response_model = None
        query_columns = None
        response_columns = None

        if signature == QUERY:
            query_config = self._query_config
            query_model = self._query_model
            query_columns = self._query if self._query else list(data.columns)
            if isinstance(query_columns, tuple):
                query_columns = query_columns[0]
        elif signature == RESPONSE:
            response_config = self._response_config
            response_model = self._response_model
            response_columns = self._response if self._response else list(data.columns)
            if isinstance(response_columns, tuple):
                response_columns = response_columns[0]
        else:
            query_config = self._query_config
            query_model = self._query_model
            response_config = self._response_config
            response_model = self._response_model
            assert self._query and self._response
            query_columns = self._query
            response_columns = self._response

        logger.debug(f"signature: {signature}")
        logger.debug(f"column_types: {column_types}")
        logger.debug(f"query_columns: {query_columns}")
        logger.debug(f"response_columns: {response_columns}")
        query_df_preprocessor, response_df_preprocessor, label_df_preprocessor = self._get_matcher_df_preprocessor(
            data=data,
            column_types=column_types,
            query_config=query_config,
            response_config=response_config,
            query_columns=query_columns,
            response_columns=response_columns,
        )

        query_processors, response_processors, label_processors = self._get_matcher_data_processors(
            query_model=query_model,
            query_config=query_config,
            response_model=response_model,
            response_config=response_config,
        )

        logger.debug(f"query_processors: {query_processors}")
        logger.debug(f"response_processors: {response_processors}")
        logger.debug(f"label_processors: {label_processors}")

        # For prediction data with no labels provided.
        df_preprocessors = [query_df_preprocessor, response_df_preprocessor]
        data_processors = [query_processors, response_processors]
        if requires_label:
            df_preprocessors.append(label_df_preprocessor)
            data_processors.append(label_processors)

        df_preprocessors = [item for item in df_preprocessors if item is not None]
        data_processors = [item for item in data_processors if item is not None]

        if self._match_label is not None:
            match_label = label_df_preprocessor.label_generator.transform([self._match_label]).item()
        else:
            match_label = None

        return data, df_preprocessors, data_processors, match_label

    def _default_predict(
        self,
        data: Union[pd.DataFrame, Dict, List],
        id_mappings: Union[Dict[str, Dict], Dict[str, pd.Series]],
        df_preprocessor: List[MultiModalFeaturePreprocessor],
        data_processors: List[Dict],
        num_gpus: int,
        precision: Union[int, str],
        batch_size: int,
        strategy: str,
        match_label: int,
        signature: Optional[str] = None,
        barebones: Optional[bool] = False,
    ) -> List[Dict]:
        datamodule = BaseDataModule(
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
            per_gpu_batch_size=batch_size,
            num_workers=self._config.env.num_workers_inference,
            predict_data=data,
            id_mappings=id_mappings,
        )
        litmodule = MatcherLitModule(
            query_model=self._query_model,
            response_model=self._response_model,
            signature=signature,
            match_label=match_label,
        )
        pred_writer = self.get_pred_writer(strategy=strategy)
        callbacks = self.get_callbacks_per_run(pred_writer=pred_writer, is_train=False)
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
        self.clean_trainer_processes(trainer=trainer, is_train=False)

        return outputs

    def _realtime_predict(
        self,
        data: pd.DataFrame,
        df_preprocessor: Union[MultiModalFeaturePreprocessor, List[MultiModalFeaturePreprocessor]],
        data_processors: Union[Dict, List[Dict]],
        num_gpus: int,
        precision: Union[int, str],
        query_model: Optional[nn.Module] = None,
        response_model: Optional[nn.Module] = None,
        id_mappings: Union[Dict[str, Dict], Dict[str, pd.Series]] = None,
        signature: Optional[str] = None,
        match_label: Optional[int] = None,
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
        query_model
            Matcher's query model.
        response_model
            Matcher's response model.
        id_mappings
            Id-to-content mappings. The contents can be text, image, etc.
            This is used when the dataframe contains the query/response indexes instead of their contents.
        signature
            query or response.
        match_label
            0 or 1.

        Returns
        -------
        A list of output dicts.
        """

        batch = self.process_batch(
            data=data,
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
            id_mappings=id_mappings,
        )
        output = self.predict_matcher_batch(
            batch=batch,
            query_model=query_model,
            response_model=response_model,
            signature=signature,
            match_label=match_label,
            precision=precision,
            num_gpus=num_gpus,
        )

        return [output]

    def predict_per_run(
        self,
        data: Union[pd.DataFrame, dict, list],
        realtime: Optional[bool],
        requires_label: bool,
        id_mappings: Union[Dict[str, Dict], Dict[str, pd.Series]] = None,
        signature: Optional[str] = None,
        barebones: Optional[bool] = False,
    ) -> List[Dict]:
        """
        Perform inference for predictor or matcher.

        Parameters
        ----------
        data
            The data for inference.
        realtime
            Whether use realtime inference.
        requires_label
            Whether uses label during inference.
        id_mappings
            Id-to-content mappings. The contents can be text, image, etc.
            This is used when the dataframe contains the query/response indexes instead of their contents.
        signature
            query or response.
        barebones
            Whether to run in “barebones mode”, where all lightning's features that may impact raw speed are disabled.

        Returns
        -------
        A list of output dicts.
        """
        data, df_preprocessor, data_processors, match_label = self._on_predict_start(
            data=data,
            id_mappings=id_mappings,
            requires_label=requires_label,
            signature=signature,
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
        batch_size = self.get_predict_batch_size_per_run(
            num_gpus=num_gpus,
            strategy=strategy,
        )
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
            outputs = self._realtime_predict(
                query_model=self._query_model,
                response_model=self._response_model,
                data=data,
                df_preprocessor=df_preprocessor,
                data_processors=data_processors,
                num_gpus=num_gpus,
                precision=precision,
                id_mappings=id_mappings,
                signature=signature,
                match_label=match_label,
            )
        else:
            outputs = self._default_predict(
                data=data,
                id_mappings=id_mappings,
                df_preprocessor=df_preprocessor,
                data_processors=data_processors,
                num_gpus=num_gpus,
                precision=precision,
                batch_size=batch_size,
                strategy=strategy,
                match_label=match_label,
                signature=signature,
            )

        return outputs

    def _evaluate_symmetric_ranking(self, data):
        data_with_label, query_data, response_data, label_column = convert_data_for_ranking(
            data=data,
            query_column=self._query[0],
            response_column=self._response[0],
        )
        logger.debug(f"first _evaluate_ranking...\n")
        score_1 = self._evaluate_ranking(
            qr_relevance=data_with_label,
            query_data=query_data,
            response_data=response_data,
            label_column=label_column,
            cutoffs=[1, 5, 10],
        )
        data_with_label, query_data, response_data, label_column = convert_data_for_ranking(
            data=data,
            query_column=self._response[0],
            response_column=self._query[0],
        )
        logger.debug(f"second _evaluate_ranking...\n")
        score_2 = self._evaluate_ranking(
            qr_relevance=data_with_label,
            query_data=query_data,
            response_data=response_data,
            label_column=label_column,
            cutoffs=[1, 5, 10],
        )

        return sum(score_1.values()) + sum(score_2.values())

    def _evaluate_ranking(
        self,
        qr_relevance: Union[pd.DataFrame, dict, list],
        query_data: Union[pd.DataFrame, dict, list],
        response_data: Union[pd.DataFrame, dict, list],
        label_column: str,
        id_mappings: Optional[Union[Dict[str, Dict], Dict[str, pd.Series]]] = None,
        metrics: Optional[Union[str, List[str]]] = None,
        chunk_size: Optional[int] = 1024,
        similarity_type: Optional[str] = "cosine",
        cutoffs: Optional[List[int]] = [1, 5, 10],
        realtime: Optional[bool] = False,
    ):
        query_column = query_data.columns[0]
        response_column = response_data.columns[0]

        qr_relevance = data_to_df(data=qr_relevance)
        assert query_column in qr_relevance.columns
        assert response_column in qr_relevance.columns

        if metrics is None:
            metrics = [self._eval_metric_name]
        if isinstance(metrics, str):
            metrics = [metrics]

        rank_labels = {}
        for i, per_row in qr_relevance.iterrows():
            rank_labels.setdefault(per_row[query_column], {})[per_row[response_column]] = int(per_row[label_column])

        rank_results = dict()
        query_embeddings = self.extract_embedding(
            query_data, id_mappings=id_mappings, as_tensor=True, realtime=realtime
        )
        num_chunks = max(1, len(response_data) // chunk_size)
        top_k = max(cutoffs)
        for response_chunk in np.array_split(response_data, num_chunks):
            response_embeddings = self.extract_embedding(
                response_chunk, id_mappings=id_mappings, as_tensor=True, realtime=realtime
            )
            similarity_scores = compute_semantic_similarity(
                a=query_embeddings, b=response_embeddings, similarity_type=similarity_type
            )
            similarity_scores[torch.isnan(similarity_scores)] = -1
            top_k_scores, top_k_indices = torch.topk(
                similarity_scores,
                k=min(top_k + 1, len(similarity_scores[1])),
                dim=1,
                largest=True,
                sorted=False,
            )
            top_k_indices = top_k_indices.cpu().tolist()
            top_k_scores = top_k_scores.cpu().tolist()
            for i in range(len(query_data)):
                query_idx = query_data.iloc[i][query_column]
                for sub_response_idx, score in zip(top_k_indices[i], top_k_scores[i]):
                    response_idx = response_chunk.iloc[int(sub_response_idx)][response_column]
                    rank_results.setdefault(query_idx, {})[response_idx] = score

        results = compute_ranking_score(results=rank_results, qrel_dict=rank_labels, metrics=metrics, cutoffs=cutoffs)

        return results

    def _evaluate_matching(
        self,
        data: Union[pd.DataFrame, dict, list],
        id_mappings: Optional[Union[Dict[str, Dict], Dict[str, pd.Series]]] = None,
        metrics: Optional[Union[str, List[str]]] = None,
        return_pred: Optional[bool] = False,
        realtime: Optional[bool] = False,
    ):
        outputs = self.predict_per_run(
            data=data,
            id_mappings=id_mappings,
            realtime=realtime,
            requires_label=True,
        )
        prob = extract_from_output(ret_type=PROBABILITY, outputs=outputs)

        metric_data = {Y_PRED_PROB: prob}

        y_pred = self._label_df_preprocessor.transform_prediction(
            y_pred=prob,
            inverse_categorical=False,
        )
        y_pred_inv = self._label_df_preprocessor.transform_prediction(
            y_pred=prob,
            inverse_categorical=True,
        )
        y_true = self._label_df_preprocessor.transform_label_for_metric(df=data)

        metric_data.update(
            {
                Y_PRED: y_pred,
                Y_TRUE: y_true,
            }
        )

        if metrics is None:
            metrics = [self._eval_metric_name]
        if isinstance(metrics, str):
            metrics = [metrics]

        results = {}
        for per_metric in metrics:
            score = compute_score(
                metric_data=metric_data,
                metric=per_metric.lower(),
            )
            results[per_metric] = score

        if return_pred:
            return results, self._as_pandas(data=data, to_be_converted=y_pred_inv)
        else:
            return results

    def evaluate(
        self,
        data: Union[pd.DataFrame, dict, list],
        query_data: Optional[pd.DataFrame, dict, list] = None,
        response_data: Optional[pd.DataFrame, dict, list] = None,
        id_mappings: Optional[Union[Dict[str, Dict], Dict[str, pd.Series]]] = None,
        metrics: Optional[Union[str, List[str]]] = None,
        return_pred: Optional[bool] = False,
        chunk_size: Optional[int] = 1024,
        similarity_type: Optional[str] = "cosine",
        cutoffs: Optional[List[int]] = [1, 5, 10],
        label: Optional[str] = None,
        realtime: Optional[bool] = False,
        **kwargs,
    ):
        """
        Evaluate model on a test dataset.

        Parameters
        ----------
        data
            A dataframe, containing the query data, response data, and their relevance. For example,
            | query_col1  | query_col2 | response_col1 | response_col2 | relevance_score |
            |-------------|------------|---------------|---------------|-----------------|
            |             | ....       | ....          | ...           | ...             |
            |             | ....       | ....          | ...           | ...             |
        query_data
            Query data used for ranking.
        response_data
            Response data used for ranking.
        id_mappings
             Id-to-content mappings. The contents can be text, image, etc.
             This is used when data/query_data/response_data contain the query/response identifiers instead of their contents.
        metrics
            A list of metric names to report.
            If None, we only return the score for the stored `_eval_metric_name`.
        return_pred
            Whether to return the prediction result of each row.
        chunk_size
            Scan the response data by chunk_size each time. Increasing the value increases the speed, but requires more memory.
        similarity_type
            Use what function (cosine/dot_prod) to score the similarity (default: cosine).
        cutoffs
            A list of cutoff values to evaluate ranking.
        label
            The label column name in data. Some tasks, e.g., image<-->text matching, have no label column in training data,
            but the label column is still required in evaluation.
        realtime
            Whether to do realtime inference, which is efficient for small data (default False).
            If provided None, we would infer it on based on the data modalities
            and sample number.

        Returns
        -------
        A dictionary with the metric names and their corresponding scores.
        Optionally return a dataframe of prediction results.
        """
        self._ensure_inference_ready()
        if all(v is not None for v in [data, query_data, response_data]):
            if isinstance(query_data, list):
                assert self._query is not None, (
                    "query_data is a list. Need a dict or dataframe, whose keys or headers should be in data's headers."
                )

            if isinstance(response_data, list):
                assert self._response is not None, (
                    "response_data is a list. Need a dict or dataframe, whose keys or headers should be in data's headers."
                )

            query_header = self._query[0] if self._query else None
            query_data = data_to_df(data=query_data, header=query_header)

            response_header = self._response[0] if self._response else None
            response_data = data_to_df(data=response_data, header=response_header)

            if label is None:
                label = self._label_column

            return self._evaluate_ranking(
                qr_relevance=data,
                query_data=query_data,
                response_data=response_data,
                label_column=label,
                id_mappings=id_mappings,
                metrics=metrics,
                chunk_size=chunk_size,
                similarity_type=similarity_type,
                cutoffs=cutoffs,
                realtime=realtime,
            )
        elif data is not None:
            return self._evaluate_matching(
                data=data,
                id_mappings=id_mappings,
                metrics=metrics,
                return_pred=return_pred,
                realtime=realtime,
            )
        else:
            raise ValueError(f"Invalid input.")

    def predict(
        self,
        data: Union[pd.DataFrame, dict, list],
        id_mappings: Optional[Union[Dict[str, Dict], Dict[str, pd.Series]]] = None,
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
        id_mappings
             Id-to-content mappings. The contents can be text, image, etc.
             This is used when data contain the query/response identifiers instead of their contents.
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
        self._ensure_inference_ready()
        outputs = self.predict_per_run(
            data=data,
            id_mappings=id_mappings,
            realtime=realtime,
            requires_label=False,
        )
        prob = extract_from_output(outputs=outputs, ret_type=PROBABILITY)

        if self._label_df_preprocessor:
            pred = self._label_df_preprocessor.transform_prediction(
                y_pred=prob,
            )
        else:
            if isinstance(prob, (torch.Tensor, np.ndarray)) and prob.ndim == 2:
                pred = prob.argmax(axis=1)
            else:
                pred = prob

        if (as_pandas is None and isinstance(data, pd.DataFrame)) or as_pandas is True:
            pred = self._as_pandas(data=data, to_be_converted=pred)

        return pred

    def predict_proba(
        self,
        data: Union[pd.DataFrame, dict, list],
        id_mappings: Optional[Union[Dict[str, Dict], Dict[str, pd.Series]]] = None,
        as_pandas: Optional[bool] = None,
        as_multiclass: Optional[bool] = True,
        realtime: Optional[bool] = False,
        **kwargs,
    ):
        """
        Predict probabilities class probabilities rather than class labels.
        This is only for the classification tasks. Calling it for a regression task will throw an exception.

        Parameters
        ----------
        data
            The data to make predictions for. Should contain same column names as training data and
              follow same format (except for the `label` column).
        id_mappings
             Id-to-content mappings. The contents can be text, image, etc.
             This is used when data contain the query/response identifiers instead of their contents.
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
        self._ensure_inference_ready()
        outputs = self.predict_per_run(
            data=data,
            id_mappings=id_mappings,
            realtime=realtime,
            requires_label=False,
        )
        prob = extract_from_output(outputs=outputs, ret_type=PROBABILITY)

        if not as_multiclass:
            if self._problem_type == BINARY:
                prob = prob[:, 1]

        if (as_pandas is None and isinstance(data, pd.DataFrame)) or as_pandas is True:
            prob = self._as_pandas(data=data, to_be_converted=prob)

        return prob

    def extract_embedding(
        self,
        data: Union[pd.DataFrame, dict, list],
        signature: Optional[str] = None,
        id_mappings: Optional[Union[Dict[str, Dict], Dict[str, pd.Series]]] = None,
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
        signature
            query or response
        id_mappings
             Id-to-content mappings. The contents can be text, image, etc.
             This is used when data contain the query/response identifiers instead of their contents.
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
        self._ensure_inference_ready()
        if signature is None:
            if self._query or self._response:
                if isinstance(data, list):
                    raise ValueError("data can't be a list. Provide a dict or a dataframe instead.")
                else:
                    data = data_to_df(data=data)
                    if self._query and all(c in data.columns for c in self._query):
                        signature = QUERY
                    elif self._response and all(c in data.columns for c in self._response):
                        signature = RESPONSE
                    else:
                        raise ValueError(
                            f"Both query `{self._query}` and response `{self._response}` are not within the data headers `{data.columns}`."
                        )
            else:
                signature = QUERY

        outputs = self.predict_per_run(
            data=data,
            id_mappings=id_mappings,
            signature=signature,
            realtime=realtime,
            requires_label=False,
        )
        features = extract_from_output(outputs=outputs, ret_type=FEATURES, as_ndarray=as_tensor is False)

        if as_pandas:
            features = pd.DataFrame(features, index=data.index)

        return features

    def _as_pandas(
        self,
        data: Union[pd.DataFrame, dict, list],
        to_be_converted: np.ndarray,
    ):
        if isinstance(data, pd.DataFrame):
            index = data.index
        else:
            index = None
        if to_be_converted.ndim == 1:
            return pd.Series(to_be_converted, index=index, name=self._label_column)
        else:
            return pd.DataFrame(to_be_converted, index=index, columns=self.class_labels)

    def _load_state_dict(
        self,
        state_dict: dict = None,
        path: str = None,
        query_prefix: str = "query_model.",
        response_prefix: str = "response_model.",
    ):
        if state_dict is None:
            state_dict = torch.load(path, map_location=torch.device("cpu"))["state_dict"]  # nosec B614
        query_state_dict = {
            k.partition(query_prefix)[2]: v for k, v in state_dict.items() if k.startswith(query_prefix)
        }
        self._query_model.load_state_dict(query_state_dict)

        response_state_dict = {
            k.partition(response_prefix)[2]: v for k, v in state_dict.items() if k.startswith(response_prefix)
        }
        self._response_model.load_state_dict(response_state_dict)

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
        query_config: Optional[DictConfig] = None,
        response_config: Optional[DictConfig] = None,
        query_model: Optional[nn.Module] = None,
        response_model: Optional[nn.Module] = None,
        query_df_preprocessor: Optional[MultiModalFeaturePreprocessor] = None,
        response_df_preprocessor: Optional[MultiModalFeaturePreprocessor] = None,
        label_df_preprocessor: Optional[MultiModalFeaturePreprocessor] = None,
        query_processors: Optional[Dict] = None,
        response_processors: Optional[Dict] = None,
        label_processors: Optional[Dict] = None,
        fit_called: Optional[bool] = None,
        save_model: Optional[bool] = True,
    ):
        """
        Save this matcher to file in directory specified by `path`.

        Parameters
        ----------
        path
            The directory to save this matcher.
        standalone
            Whether to save the downloaded model for offline deployment.
            When standalone = True, save the transformers.CLIPModel and transformers.AutoModel to os.path.join(path,model_name),
            and reset the associate model.model_name.checkpoint_name start with `local://` in config.yaml.
            When standalone = False, the saved artifact may require an online environment to process in load().
        """
        config = config if config else self._config
        query_config = query_config if query_config else self._query_config
        response_config = response_config if response_config else self._response_config
        query_model = query_model if query_model else self._query_model
        response_model = response_model if response_model else self._response_model
        query_df_preprocessor = query_df_preprocessor if query_df_preprocessor else self._query_df_preprocessor
        response_df_preprocessor = (
            response_df_preprocessor if response_df_preprocessor else self._response_df_preprocessor
        )
        label_df_preprocessor = label_df_preprocessor if label_df_preprocessor else self._label_df_preprocessor
        query_processors = query_processors if query_processors else self._query_processors
        response_processors = response_processors if response_processors else self._response_processors
        label_processors = label_processors if label_processors else self._label_processors

        path = os.path.abspath(os.path.expanduser(path))
        query_config = copy.deepcopy(query_config)
        response_config = copy.deepcopy(response_config)
        if standalone:
            query_config = save_pretrained_model_configs(model=query_model, config=query_config, path=path)
            response_config = save_pretrained_model_configs(model=response_model, config=response_config, path=path)

        os.makedirs(path, exist_ok=True)
        config = {"generic": config, QUERY: query_config, RESPONSE: response_config}
        OmegaConf.save(config=config, f=os.path.join(path, "config.yaml"))

        df_preprocessor = {
            QUERY: query_df_preprocessor,
            RESPONSE: response_df_preprocessor,
            LABEL: label_df_preprocessor,
        }
        with open(os.path.join(path, "df_preprocessor.pkl"), "wb") as fp:
            pickle.dump(df_preprocessor, fp)

        # Save text tokenizers before saving data processors
        query_processors = copy.deepcopy(query_processors)
        if TEXT in query_processors:
            for per_text_processor in query_processors[TEXT]:
                per_text_processor.save_tokenizer(path)

        # Save text tokenizers before saving data processors
        response_processors = copy.deepcopy(response_processors)
        if TEXT in response_processors:
            for per_text_processor in response_processors[TEXT]:
                per_text_processor.save_tokenizer(path)

        data_processors = {
            QUERY: query_processors,
            RESPONSE: response_processors,
            LABEL: label_processors,
        }
        with open(os.path.join(path, "data_processors.pkl"), "wb") as fp:
            pickle.dump(data_processors, fp)

        with open(os.path.join(path, f"assets.json"), "w") as fp:
            json.dump(
                {
                    "learner_class": self.__class__.__name__,
                    "query": self._query,
                    "response": self._response,
                    "match_label": self._match_label,
                    "column_types": self._column_types,
                    "label_column": self._label_column,
                    "problem_type": self._problem_type,
                    "pipeline": self._pipeline,
                    "presets": self._presets,
                    "eval_metric_name": self._eval_metric_name,
                    "validation_metric_name": self._validation_metric_name,
                    "minmax_mode": self._minmax_mode,
                    "output_shape": self._output_shape,
                    "save_path": self._save_path,
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
            task = MatcherLitModule(
                query_model=self._query_model,
                response_model=self._response_model,
            )
            checkpoint = {"state_dict": task.state_dict()}
            torch.save(checkpoint, os.path.join(path, MODEL_CHECKPOINT))  # nosec B614

    @staticmethod
    def _load_metadata(
        matcher: MatchingLearner,
        path: str,
        resume: Optional[bool] = False,
        verbosity: Optional[int] = 3,
    ):
        path = os.path.abspath(os.path.expanduser(path))
        assert os.path.isdir(path), f"'{path}' must be an existing directory."
        config = OmegaConf.load(os.path.join(path, "config.yaml"))
        query_config = config[QUERY]
        response_config = config[RESPONSE]
        config = config["generic"]

        query_config = get_local_pretrained_config_paths(
            config=query_config, path=path
        )  # check the config to load offline pretrained model configs

        response_config = get_local_pretrained_config_paths(
            config=response_config, path=path
        )  # check the config to load offline pretrained model configs

        with open(os.path.join(path, "assets.json"), "r") as fp:
            assets = json.load(fp)

        with open(os.path.join(path, "df_preprocessor.pkl"), "rb") as fp:
            df_preprocessor = pickle.load(fp)  # nosec B301

        query_df_preprocessor = df_preprocessor[QUERY]
        response_df_preprocessor = df_preprocessor[RESPONSE]
        label_df_preprocessor = df_preprocessor[LABEL]

        try:
            with open(os.path.join(path, "data_processors.pkl"), "rb") as fp:
                data_processors = pickle.load(fp)  # nosec B301

            query_processors = data_processors[QUERY]
            response_processors = data_processors[RESPONSE]
            label_processors = data_processors[LABEL]

            # Load text tokenizers after loading data processors.
            if TEXT in query_processors:
                for per_text_processor in query_processors[TEXT]:
                    per_text_processor.load_tokenizer(path)

            # Only keep the modalities with non-empty processors.
            query_processors = {k: v for k, v in query_processors.items() if len(v) > 0}

            # Load text tokenizers after loading data processors.
            if TEXT in response_processors:
                for per_text_processor in response_processors[TEXT]:
                    per_text_processor.load_tokenizer(path)

            # Only keep the modalities with non-empty processors.
            response_processors = {k: v for k, v in response_processors.items() if len(v) > 0}
        except:  # reconstruct the data processor in case something went wrong.
            query_processors = None
            response_processors = None
            label_processors = None

        matcher._query = assets["query"]
        matcher._response = assets["response"]
        matcher._match_label = assets["match_label"]
        matcher._label_column = assets["label_column"]
        matcher._problem_type = assets["problem_type"]
        matcher._pipeline = assets["pipeline"]
        matcher._presets = assets["presets"]
        matcher._eval_metric_name = assets["eval_metric_name"]
        matcher._verbosity = verbosity
        matcher._resume = resume
        matcher._save_path = path  # in case the original exp dir is copied to somewhere else
        matcher._pretrained_path = path
        matcher._pretrained = assets["pretrained"]
        matcher._fit_called = assets["fit_called"]
        matcher._config = config
        matcher._query_config = query_config
        matcher._response_config = response_config
        matcher._output_shape = assets["output_shape"]
        matcher._column_types = assets["column_types"]
        matcher._validation_metric_name = assets["validation_metric_name"]
        matcher._query_df_preprocessor = query_df_preprocessor
        matcher._response_df_preprocessor = response_df_preprocessor
        matcher._label_df_preprocessor = label_df_preprocessor
        matcher._query_processors = query_processors
        matcher._response_processors = response_processors
        matcher._label_processors = label_processors
        matcher._minmax_mode = assets["minmax_mode"]

        return matcher

    @classmethod
    def load(
        cls,
        path: str,
        resume: Optional[bool] = False,
        verbosity: Optional[int] = 3,
    ):
        """
        Load a matcher object from a directory specified by `path`. The to-be-loaded matcher
        can be completely or partially trained by .fit(). If a previous training has completed,
        it will load the checkpoint `model.ckpt`. Otherwise if a previous training accidentally
        collapses in the middle, it can load the `last.ckpt` checkpoint by setting `resume=True`.

        Parameters
        ----------
        path
            The directory to load the matcher object.
        resume
            Whether to resume training from `last.ckpt`. This is useful when a training was accidentally
            broken during the middle and we want to resume the training from the last saved checkpoint.
        verbosity
            Verbosity levels range from 0 to 4 and control how much information is printed.
            Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).

        Returns
        -------
        The loaded matcher object.
        """
        dir_path, ckpt_path = get_dir_ckpt_paths(path=path)

        assert os.path.isdir(dir_path), f"'{dir_path}' must be an existing directory."
        matcher = cls(query="", response="")
        matcher = cls._load_metadata(matcher=matcher, path=dir_path, resume=resume, verbosity=verbosity)
        matcher._query_model, matcher._response_model = create_siamese_model(
            query_config=matcher._query_config,
            response_config=matcher._response_config,
            pretrained=False,
        )
        load_path, ckpt_path = get_load_ckpt_paths(
            ckpt_path=ckpt_path,
            dir_path=dir_path,
            resume=resume,
        )
        matcher._load_state_dict(path=load_path)
        matcher._ckpt_path = ckpt_path

        return matcher

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
            return self._label_df_preprocessor.label_generator.classes_
        else:
            warnings.warn("Accessing class names for a non-classification problem. Return None.")
            return None
