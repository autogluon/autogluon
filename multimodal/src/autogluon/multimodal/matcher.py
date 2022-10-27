from __future__ import annotations

import copy
import json
import logging
import operator
import os
import pickle
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
from sklearn.model_selection import train_test_split
from torch import nn

from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.common.utils.utils import setup_outputdir
from autogluon.core.utils.utils import default_holdout_frac

from . import version as ag_version
from .constants import (
    AUTOMM,
    AUTOMM_TUTORIAL_MODE,
    BEST,
    BEST_K_MODELS_FILE,
    BINARY,
    CLASSIFICATION,
    FEATURES,
    GREEDY_SOUP,
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
    TRIPLET,
    UNIFORM_SOUP,
    Y_PRED,
    Y_PRED_PROB,
    Y_TRUE,
)
from .data.datamodule import BaseDataModule
from .data.infer_types import (
    infer_column_types,
    infer_label_column_type_by_problem_type_and_pipeline,
    infer_problem_type_output_shape,
)
from .optimization.lit_matcher import MatcherLitModule
from .optimization.utils import get_matcher_loss_func, get_matcher_miner_func, get_metric
from .utils import (
    AutoMMModelCheckpoint,
    CustomUnpickler,
    LogFilter,
    apply_log_filter,
    assign_feature_column_names,
    average_checkpoints,
    compute_inference_batch_size,
    compute_num_gpus,
    compute_ranking_score,
    compute_score,
    compute_semantic_similarity,
    create_fusion_data_processors,
    create_siamese_model,
    customize_model_names,
    data_to_df,
    extract_from_output,
    get_config,
    get_local_pretrained_config_paths,
    get_minmax_mode,
    infer_dtypes_by_model_names,
    infer_metrics,
    infer_precision,
    init_df_preprocessor,
    init_pretrained_matcher,
    load_text_tokenizers,
    process_save_path,
    save_pretrained_model_configs,
    save_text_tokenizers,
    select_model,
    tensor_to_ndarray,
    try_to_infer_pos_label,
)

logger = logging.getLogger(AUTOMM)


class MultiModalMatcher:
    """
    MultiModalMatcher is a framework to learn/extract embeddings for multimodal data including image, text, and tabular.
    These embeddings can be used e.g. with cosine-similarity to find items with similar semantic meanings.
    This can be useful for computing the semantic similarity of two items, semantic search, paraphrase mining, etc.
    """

    def __init__(
        self,
        query: Optional[Union[str, List[str]]] = None,
        response: Optional[Union[str, List[str]]] = None,
        negative: Optional[Union[str, List[str]]] = None,
        label: Optional[str] = None,
        match_label: Optional[Union[int, str]] = None,
        problem_type: Optional[str] = None,
        pipeline: Optional[str] = None,
        eval_metric: Optional[str] = None,
        hyperparameters: Optional[dict] = None,
        path: Optional[str] = None,
        verbosity: Optional[int] = 3,
        warn_if_exist: Optional[bool] = True,
        enable_progress_bar: Optional[bool] = None,
    ):
        """
        Parameters
        ----------
        query
            Names of columns that contains the query data.
        response
            Names of columns that contains the response data. If no label column is provided,
            query and response columns form positive pairs.
        negative
            Names of columns that contains the negative data. Query and negative make up negative pairs.
        label
            Name of the label column. Label and negative shouldn't be used simultaneously.
        match_label
            For binary labels, which label indicates a pair of items match.
        problem_type
            Type of matching problem if the label column is available.
            This could be binary, multiclass, or regression
            if the label column contains binary, multiclass, or numeric labels.
            If `problem_type = None`, the prediction problem type is inferred
            based on the label-values in provided dataset.
        eval_metric
            Evaluation metric name. If `eval_metric = None`, it is automatically chosen based on `problem_type`.
            Defaults to 'accuracy' for binary and multiclass classification, 'root_mean_squared_error' for regression.
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

        if os.environ.get(AUTOMM_TUTORIAL_MODE):
            verbosity = 1  # don't use 3, which doesn't suppress logger.info() in .load().
            enable_progress_bar = False

        if verbosity is not None:
            set_logger_verbosity(verbosity, logger=logger)

        if path is not None:
            path = process_save_path(path=path)

        if isinstance(query, str):
            query = [query]
        if query:
            assert all(isinstance(q, str) for q in query)

        if isinstance(response, str):
            response = [response]
        if response:
            assert all(isinstance(r, str) for r in response)

        if isinstance(negative, str):
            negative = [negative]
        if negative:
            assert all(isinstance(n, str) for n in negative)
            data_format = TRIPLET
        else:
            data_format = PAIR

        self._query = query
        self._response = response
        self._negative = negative
        self._data_format = data_format
        self._match_label = match_label
        self._label_column = label
        self._problem_type = problem_type.lower() if problem_type is not None else None
        self._pipeline = pipeline.lower() if pipeline is not None else None
        self._eval_metric_name = eval_metric
        self._validation_metric_name = None
        self._output_shape = None
        self._save_path = path
        self._ckpt_path = None
        self._pretrained_path = None
        self._config = None
        self._query_config = None
        self._response_config = None
        self._query_df_preprocessor = None
        self._response_df_preprocessor = None
        self._negative_df_preprocessor = None
        self._label_df_preprocessor = None
        self._column_types = None
        self._query_processors = None
        self._response_processors = None
        self._negative_processors = None
        self._label_processors = None
        self._query_model = None
        self._response_model = None
        self._resume = False
        self._continuous_training = False
        self._verbosity = verbosity
        self._warn_if_exist = warn_if_exist
        self._enable_progress_bar = enable_progress_bar if enable_progress_bar is not None else True

        if self._pipeline is not None:
            (
                self._config,
                self._query_config,
                self._response_config,
                self._query_model,
                self._response_model,
                self._query_processors,
                self._response_processors,
            ) = init_pretrained_matcher(pipeline=self._pipeline, hyperparameters=hyperparameters)

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
        return self._problem_type

    @property
    def column_types(self):
        return self._column_types

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

    def fit(
        self,
        train_data: pd.DataFrame,
        id_mappings: Optional[
            Dict[str, Dict]
        ] = None,  # TODO rename it to linked_content and support dict of pd.Series.
        presets: Optional[str] = None,
        tuning_data: Optional[pd.DataFrame] = None,
        time_limit: Optional[int] = None,
        save_path: Optional[str] = None,
        hyperparameters: Optional[Union[str, Dict, List[str]]] = None,
        column_types: Optional[dict] = None,
        holdout_frac: Optional[float] = None,
        seed: Optional[int] = 123,
    ):
        """
        Fit MultiModalMatcher. Train the model to learn embeddings to simultaneously maximize and minimize
        the semantic similarities of positive and negative pairs.
        The data may contain image, text, numeric, or categorical features.

        Parameters
        ----------
        train_data
            A dataframe, containing the query data, response data, and their relevance. For example,
            | query_col1  | query_col2 | response_col1 | response_col2 | relevance_score |
            |-------------|------------|---------------|---------------|-----------------|
            |             | ....       | ....          | ...           | ...             |
            |             | ....       | ....          | ...           | ...             |
        id_mappings
             Id-to-content mappings. The contents can be text, image, etc.
             This is used when the dataframe contains the query/response indexes instead of their contents.
        presets
            Name of the presets. See the available presets in `presets.py`.
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
        An "MultiModalMatcher" object (itself).
        """

        pl.seed_everything(seed, workers=True)

        if self._resume:
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
            id_mappings=id_mappings,
        )
        column_types = infer_label_column_type_by_problem_type_and_pipeline(
            column_types=column_types,
            label_columns=self._label_column,
            problem_type=self._problem_type,
            data=train_data,
            valid_data=tuning_data,
        )
        problem_type, output_shape = infer_problem_type_output_shape(
            label_column=self._label_column,
            column_types=column_types,
            data=train_data,
            provided_problem_type=self._problem_type,
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

        if self._output_shape is not None:
            assert self._output_shape == output_shape, (
                f"Inferred output shape {output_shape} is different from " f"the previous {self._output_shape}"
            )

        if self._validation_metric_name is None or self._eval_metric_name is None:
            validation_metric_name, eval_metric_name = infer_metrics(
                problem_type=problem_type,
                eval_metric_name=self._eval_metric_name,
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
        self._output_shape = output_shape
        self._column_types = column_types

        _fit_args = dict(
            train_df=train_data,
            val_df=tuning_data,
            id_mappings=id_mappings,
            validation_metric_name=validation_metric_name,
            minmax_mode=minmax_mode,
            max_time=time_limit,
            save_path=save_path,
            ckpt_path=self._ckpt_path,
            resume=self._resume,
            enable_progress_bar=self._enable_progress_bar,
            presets=presets,
            hyperparameters=hyperparameters,
        )

        self._fit(**_fit_args)
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
        if self._query_df_preprocessor is None and query_config is not None:
            query_df_preprocessor = init_df_preprocessor(
                config=query_config,
                column_types={k: column_types[k] for k in query_columns},
                train_df_x=data[query_columns],
            )
        else:  # continuing training
            query_df_preprocessor = self._query_df_preprocessor

        if self._response_df_preprocessor is None and response_config is not None:
            response_df_preprocessor = init_df_preprocessor(
                config=response_config,
                column_types={k: column_types[k] for k in response_columns},
                train_df_x=data[response_columns],
            )
        else:  # continuing training
            response_df_preprocessor = self._response_df_preprocessor

        if self._label_df_preprocessor is None and response_config is not None and self._label_column in column_types:
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
    ):
        if query_model is None:
            query_processors = None
        elif self._query_processors is None and all(v is not None for v in [query_model, query_config]):
            query_processors = create_fusion_data_processors(
                model=query_model,
                config=query_config,
                requires_label=False,
                requires_data=True,
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

    def _fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        id_mappings: Dict[str, Dict],
        validation_metric_name: str,
        minmax_mode: str,
        max_time: timedelta,
        save_path: str,
        ckpt_path: str,
        resume: bool,
        enable_progress_bar: bool,
        presets: Optional[str] = None,
        hyperparameters: Optional[Union[str, Dict, List[str]]] = None,
        hpo_mode: bool = False,
        **hpo_kwargs,
    ):
        if presets == None:
            presets = "siamese_network"

        if presets == "siamese_network":
            if self._config is None:
                config = get_config(
                    presets=presets,
                    overrides=hyperparameters,
                    extra=["matcher"],
                )
            else:
                config = self._config

            if self._query_config is None:
                query_config = copy.deepcopy(config)
                # customize config model names to make them consistent with model prefixes.
                query_config.model = customize_model_names(
                    config=query_config.model, customized_names=[f"{n}_{QUERY}" for n in query_config.model.names]
                )
            else:
                query_config = self._query_config

            if self._response_config is None:
                response_config = copy.deepcopy(config)
                # customize config model names to make them consistent with model prefixes.
                response_config.model = customize_model_names(
                    config=response_config.model,
                    customized_names=[f"{n}_{RESPONSE}" for n in response_config.model.names],
                )
            else:
                response_config = self._response_config
        else:
            raise ValueError("Currently only support presets: siamese_network.")

        query_df_preprocessor, response_df_preprocessor, label_df_preprocessor = self._get_matcher_df_preprocessor(
            data=train_df,
            column_types=self._column_types,
            query_config=query_config,
            response_config=response_config,
            query_columns=self._query,
            response_columns=self._response,
        )

        query_config = select_model(config=query_config, df_preprocessor=query_df_preprocessor)
        response_config = select_model(config=response_config, df_preprocessor=response_df_preprocessor)

        if self._query_model is None or self._response_model is None:
            if presets == "siamese_network":
                query_model, response_model = create_siamese_model(
                    query_config=query_config,
                    response_config=response_config,
                )
            else:
                raise ValueError("Only support preset `siamese_network` currently.")
        else:  # continuing training
            query_model = self._query_model
            response_model = self._response_model

        query_processors, response_processors, label_processors = self._get_matcher_data_processors(
            query_model=query_model,
            query_config=query_config,
            response_model=response_model,
            response_config=response_config,
        )

        query_processors_count = {k: len(v) for k, v in query_processors.items()}
        logger.debug(f"query_processors_count: {query_processors_count}")
        response_processors_count = {k: len(v) for k, v in response_processors.items()}
        logger.debug(f"response_processors_count: {response_processors_count}")
        label_processors_count = {k: len(v) for k, v in label_processors.items()}
        logger.debug(f"label_processors_count: {label_processors_count}")

        pos_label = try_to_infer_pos_label(
            data_config=response_config.data,
            label_encoder=label_df_preprocessor.label_generator,
            problem_type=self._problem_type,
        )
        validation_metric, custom_metric_func = get_metric(
            metric_name=validation_metric_name,
            num_classes=self._output_shape,
            pos_label=pos_label,
        )

        loss_func = get_matcher_loss_func(
            data_format=self._data_format,
            problem_type=self._problem_type,
            loss_type=config.matcher.loss.type,
            pos_margin=config.matcher.loss.pos_margin,
            neg_margin=config.matcher.loss.neg_margin,
            distance_type=config.matcher.distance.type,
        )

        miner_func = get_matcher_miner_func(
            miner_type=config.matcher.miner.type,
            pos_margin=config.matcher.miner.pos_margin,
            neg_margin=config.matcher.miner.neg_margin,
            distance_type=config.matcher.distance.type,
        )

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
        self._loss_func = loss_func

        train_dm = BaseDataModule(
            df_preprocessor=[query_df_preprocessor, response_df_preprocessor, label_df_preprocessor],
            data_processors=[query_processors, response_processors, label_processors],
            per_gpu_batch_size=config.env.per_gpu_batch_size,
            num_workers=config.env.num_workers,
            train_data=train_df,
            val_data=val_df,
            id_mappings=id_mappings,
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

        if self._match_label is not None:
            match_label = label_df_preprocessor.label_generator.transform([self._match_label]).item()
        else:
            match_label = None

        task = MatcherLitModule(
            query_model=query_model,
            response_model=response_model,
            match_label=match_label,
            loss_func=loss_func,
            miner_func=miner_func,
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
                strategy = None
            else:
                strategy = config.env.strategy
        else:
            strategy = None
            num_gpus = min(num_gpus, 1)

        config.env.num_gpus = num_gpus
        config.env.precision = precision
        config.env.strategy = strategy
        self._config = config
        # save artifacts for the current running, except for model checkpoint, which will be saved in trainer
        self.save(save_path)

        blacklist_msgs = ["already configured with model summary"]
        log_filter = LogFilter(blacklist_msgs)
        with apply_log_filter(log_filter):
            trainer = pl.Trainer(
                gpus=num_gpus,
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
                    query_model=query_model,
                    response_model=response_model,
                    save_path=save_path,
                    minmax_mode=minmax_mode,
                    top_k_average_method=config.optimization.top_k_average_method,
                    val_df=val_df,
                    validation_metric_name=validation_metric_name,
                )
        else:
            sys.exit(f"Training finished, exit the process with global_rank={trainer.global_rank}...")

    def _top_k_average(
        self,
        query_model,
        response_model,
        save_path,
        minmax_mode,
        top_k_average_method,
        val_df,
        validation_metric_name,
        last_ckpt_path=None,
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
                        reverse=(minmax_mode == MAX),
                    )
                ]
                if top_k_average_method == GREEDY_SOUP:
                    # Select the ingredients based on the methods proposed in paper
                    #  "Model soups: averaging weights of multiple fine-tuned models improves accuracy without
                    #  increasing inference time", https://arxiv.org/pdf/2203.05482.pdf
                    monitor_op = {MIN: operator.le, MAX: operator.ge}[minmax_mode]

                    logger.info(f"Start to fuse {len(top_k_model_paths)} checkpoints via the greedy soup algorithm.")

                    ingredients = [top_k_model_paths[0]]
                    self._query_model, self._response_model = self._load_state_dict(
                        query_model=query_model,
                        response_model=response_model,
                        path=top_k_model_paths[0],
                    )
                    best_score = self.evaluate(val_df, metrics=[validation_metric_name])[validation_metric_name]
                    for i in range(1, len(top_k_model_paths)):
                        cand_avg_state_dict = average_checkpoints(
                            checkpoint_paths=ingredients + [top_k_model_paths[i]],
                        )
                        self._query_model, self._response_model = self._load_state_dict(
                            query_model=query_model,
                            response_model=response_model,
                            state_dict=cand_avg_state_dict,
                        )
                        cand_score = self.evaluate(val_df, metrics=[validation_metric_name])[validation_metric_name]
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
        self._query_model, self._response_model = self._load_state_dict(
            query_model=query_model,
            response_model=response_model,
            state_dict=avg_state_dict,
        )

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

    def _predict(
        self,
        data: Union[pd.DataFrame, dict, list],
        id_mappings: Dict[str, Dict],
        requires_label: bool,
        signature: Optional[str] = None,
    ) -> List[Dict]:
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
                allowable_column_types=allowable_dtypes,
                fallback_column_type=fallback_dtype,
                id_mappings=id_mappings,
            )
            if self._label_column and self._label_column in data.columns:
                column_types = infer_label_column_type_by_problem_type(
                    column_types=column_types,
                    label_columns=self._label_column,
                    problem_type=self._problem_type,
                    data=data,
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
            query_columns = (self._query if self._query else list(data.columns),)
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

        # For prediction data with no labels provided.
        df_preprocessors = [query_df_preprocessor, response_df_preprocessor]
        data_processors = [query_processors, response_processors]
        if requires_label:
            df_preprocessors.append(label_df_preprocessor)
            data_processors.append(label_processors)

        df_preprocessors = [item for item in df_preprocessors if item is not None]
        data_processors = [item for item in data_processors if item is not None]
        strategy = "dp"  # default used in inference.

        num_gpus = compute_num_gpus(config_num_gpus=self._config.env.num_gpus, strategy="dp")
        if num_gpus == 1:
            strategy = None

        precision = infer_precision(num_gpus=num_gpus, precision=self._config.env.precision)

        batch_size = compute_inference_batch_size(
            per_gpu_batch_size=self._config.env.per_gpu_batch_size,
            eval_batch_size_ratio=OmegaConf.select(self._config, "env.eval_batch_size_ratio"),
            per_gpu_batch_size_evaluation=self._config.env.per_gpu_batch_size_evaluation,  # backward compatibility.
            num_gpus=num_gpus,
            strategy=strategy,
        )

        predict_dm = BaseDataModule(
            df_preprocessor=df_preprocessors,
            data_processors=data_processors,
            per_gpu_batch_size=batch_size,
            num_workers=self._config.env.num_workers_evaluation,
            predict_data=data,
            id_mappings=id_mappings,
        )
        if self._match_label is not None:
            match_label = label_df_preprocessor.label_generator.transform([self._match_label]).item()
        else:
            match_label = None

        task = MatcherLitModule(
            query_model=query_model,
            response_model=response_model,
            signature=signature,
            match_label=match_label,
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
                gpus=num_gpus,
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

    def _evaluate_ranking(
        self,
        qr_relevance: Union[pd.DataFrame, dict, list],
        query_data: Union[pd.DataFrame, dict, list],
        response_data: Union[pd.DataFrame, dict, list],
        id_mappings: Optional[Dict[str, Dict]] = None,
        chunk_size: Optional[int] = 1024,
        similarity_type: Optional[str] = "cosine",
        top_k: Optional[int] = 100,
    ):
        query_header = self._query[0] if self._query is not None else QUERY
        query_data = data_to_df(data=query_data, header=query_header)
        response_header = self._response[0] if self._response else RESPONSE
        response_data = data_to_df(data=response_data, header=response_header)

        qr_relevance = data_to_df(data=qr_relevance)

        rank_labels = {}
        for i, per_row in qr_relevance.iterrows():
            rank_labels.setdefault(per_row[self._query[0]], {})[per_row[self._response[0]]] = int(
                per_row[self._label_column]
            )

        rank_results = dict()
        query_embeddings = self.extract_embedding(query_data, signature=QUERY, id_mappings=id_mappings, as_tensor=True)
        num_chunks = max(1, len(response_data) // chunk_size)
        for response_chunk in np.array_split(response_data, num_chunks):
            response_embeddings = self.extract_embedding(
                response_chunk, signature=RESPONSE, id_mappings=id_mappings, as_tensor=True
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
                query_idx = query_data.iloc[i][self._query[0]]
                for sub_response_idx, score in zip(top_k_indices[i], top_k_scores[i]):
                    response_idx = response_chunk.iloc[int(sub_response_idx)][self._response[0]]
                    rank_results.setdefault(query_idx, {})[response_idx] = score

        ndcg, _map, recall, precision = compute_ranking_score(results=rank_results, qrel_dict=rank_labels)

        return ndcg, _map, recall, precision

    def _evaluate_matching(
        self,
        data: Union[pd.DataFrame, dict, list],
        id_mappings: Optional[Dict[str, Dict]] = None,
        metrics: Optional[Union[str, List[str]]] = None,
        return_pred: Optional[bool] = False,
    ):
        outputs = self._predict(
            data=data,
            id_mappings=id_mappings,
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

        pos_label = try_to_infer_pos_label(
            data_config=self._response_config.data,
            label_encoder=self._label_df_preprocessor.label_generator,
            problem_type=self._problem_type,
        )
        results = {}
        for per_metric in metrics:
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

    def evaluate(
        self,
        data: Optional[Union[pd.DataFrame, dict, list]] = None,
        query_data: Optional[list] = None,
        response_data: Optional[list] = None,
        id_mappings: Optional[Dict[str, Dict]] = None,
        metrics: Optional[Union[str, List[str]]] = None,
        return_pred: Optional[bool] = False,
        chunk_size: Optional[int] = 1024,
        similarity_type: Optional[str] = "cosine",
        top_k: Optional[int] = 100,
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
            A list of queries.
        response_data
            A list of response data, in which to match the queries.
        id_mappings
             Id-to-content mappings. The contents can be text, image, etc.
             This is used when the dataframe contains the query/response indexes instead of their contents.
        metrics
            A list of metric names to report.
            If None, we only return the score for the stored `_eval_metric_name`.
        return_pred
            Whether to return the prediction result of each row.
        chunk_size
            Scan the response data by chunk_size each time. Increasing the value increases the speed, but requires more memory.
        similarity_type
            Use what function (cosine/dot_prod) to score the similarity (default: cosine).
        top_k
            Retrieve top k matching entries.

        Returns
        -------
        A dictionary with the metric names and their corresponding scores.
        Optionally return a dataframe of prediction results.
        """
        if all(v is not None for v in [data, query_data, response_data]):
            return self._evaluate_ranking(
                qr_relevance=data,
                query_data=query_data,
                response_data=response_data,
                id_mappings=id_mappings,
                chunk_size=chunk_size,
                similarity_type=similarity_type,
                top_k=top_k,
            )
        elif data is not None:
            return self._evaluate_matching(
                data=data,
                id_mappings=id_mappings,
                metrics=metrics,
                return_pred=return_pred,
            )
        else:
            raise ValueError(f"Invalid input.")

    def predict(
        self,
        data: Union[pd.DataFrame, dict, list],
        id_mappings: Optional[Dict[str, Dict]] = None,
        as_pandas: Optional[bool] = None,
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
             This is used when the dataframe contains the query/response indexes instead of their contents.
        as_pandas
            Whether to return the output as a pandas DataFrame(Series) (True) or numpy array (False).

        Returns
        -------
        Array of predictions, one corresponding to each row in given dataset.
        """
        outputs = self._predict(
            data=data,
            id_mappings=id_mappings,
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
        id_mappings: Optional[Dict[str, Dict]] = None,
        as_pandas: Optional[bool] = None,
        as_multiclass: Optional[bool] = True,
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
             This is used when the dataframe contains the query/response indexes instead of their contents.
        as_pandas
            Whether to return the output as a pandas DataFrame(Series) (True) or numpy array (False).
        as_multiclass
            Whether to return the probability of all labels or
            just return the probability of the positive class for binary classification problems.

        Returns
        -------
        Array of predicted class-probabilities, corresponding to each row in the given data.
        When as_multiclass is True, the output will always have shape (#samples, #classes).
        Otherwise, the output will have shape (#samples,)
        """
        outputs = self._predict(
            data=data,
            id_mappings=id_mappings,
            requires_label=False,
        )
        prob = extract_from_output(outputs=outputs, ret_type=PROBABILITY)

        if not as_multiclass:
            if self._problem_type == BINARY:
                pos_label = try_to_infer_pos_label(
                    data_config=self._response_config.data,
                    label_encoder=self._label_df_preprocessor.label_generator,
                    problem_type=self._problem_type,
                )
                prob = prob[:, pos_label]

        if (as_pandas is None and isinstance(data, pd.DataFrame)) or as_pandas is True:
            prob = self._as_pandas(data=data, to_be_converted=prob)

        return prob

    def extract_embedding(
        self,
        data: Union[pd.DataFrame, dict, list],
        signature: Optional[str] = None,
        id_mappings: Optional[Dict[str, Dict]] = None,
        as_tensor: Optional[bool] = False,
        as_pandas: Optional[bool] = False,
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
             This is used when the dataframe contains the query/response indexes instead of their contents.
        as_tensor
            Whether to return a Pytorch tensor.
        as_pandas
            Whether to return the output as a pandas DataFrame (True) or numpy array (False).

        Returns
        -------
        Array of embeddings, corresponding to each row in the given data.
        It will have shape (#samples, D) where the embedding dimension D is determined
        by the neural network's architecture.
        """
        if signature is None:
            signature = QUERY

        outputs = self._predict(
            data=data,
            id_mappings=id_mappings,
            signature=signature,
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

    @staticmethod
    def _load_state_dict(
        query_model: nn.Module,
        response_model: nn.Module,
        state_dict: dict = None,
        path: str = None,
        query_prefix: str = "query_model.",
        response_prefix: str = "response_model.",
    ):
        if state_dict is None:
            state_dict = torch.load(path, map_location=torch.device("cpu"))["state_dict"]
        query_state_dict = {
            k.partition(query_prefix)[2]: v for k, v in state_dict.items() if k.startswith(query_prefix)
        }
        query_model.load_state_dict(query_state_dict)

        response_state_dict = {
            k.partition(response_prefix)[2]: v for k, v in state_dict.items() if k.startswith(response_prefix)
        }
        response_model.load_state_dict(response_state_dict)
        return query_model, response_model

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
        path = os.path.abspath(os.path.expanduser(path))
        query_config = copy.deepcopy(self._query_config)
        response_config = copy.deepcopy(self._response_config)
        if standalone:
            query_config = save_pretrained_model_configs(model=self._query_model, config=query_config, path=path)
            response_config = save_pretrained_model_configs(
                model=self._response_model, config=response_config, path=path
            )

        os.makedirs(path, exist_ok=True)
        config = {"generic": self._config, QUERY: query_config, RESPONSE: response_config}
        OmegaConf.save(config=config, f=os.path.join(path, "config.yaml"))

        df_preprocessor = {
            QUERY: self._query_df_preprocessor,
            RESPONSE: self._response_df_preprocessor,
            LABEL: self._label_df_preprocessor,
        }
        with open(os.path.join(path, "df_preprocessor.pkl"), "wb") as fp:
            pickle.dump(df_preprocessor, fp)

        # Save text tokenizers before saving data processors
        query_processors = copy.deepcopy(self._query_processors)
        if TEXT in query_processors:
            query_processors[TEXT] = save_text_tokenizers(
                text_processors=query_processors[TEXT],
                path=path,
            )

        # Save text tokenizers before saving data processors
        response_processors = copy.deepcopy(self._response_processors)
        if TEXT in response_processors:
            response_processors[TEXT] = save_text_tokenizers(
                text_processors=response_processors[TEXT],
                path=path,
            )

        data_processors = {
            QUERY: query_processors,
            RESPONSE: response_processors,
            LABEL: self._label_processors,
        }
        with open(os.path.join(path, "data_processors.pkl"), "wb") as fp:
            pickle.dump(data_processors, fp)

        with open(os.path.join(path, f"assets.json"), "w") as fp:
            json.dump(
                {
                    "query": self._query,
                    "response": self._response,
                    "match_label": self._match_label,
                    "column_types": self._column_types,
                    "label_column": self._label_column,
                    "problem_type": self._problem_type,
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

        task = MatcherLitModule(
            query_model=self._query_model,
            response_model=self._response_model,
        )

        checkpoint = {"state_dict": task.state_dict()}
        torch.save(checkpoint, os.path.join(path, MODEL_CHECKPOINT))

    @staticmethod
    def _load_metadata(
        matcher: MultiModalMatcher,
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
            df_preprocessor = CustomUnpickler(fp).load()

        query_df_preprocessor = df_preprocessor[QUERY]
        response_df_preprocessor = df_preprocessor[RESPONSE]
        label_df_preprocessor = df_preprocessor[LABEL]

        try:
            with open(os.path.join(path, "data_processors.pkl"), "rb") as fp:
                data_processors = CustomUnpickler(fp).load()

            query_processors = data_processors[QUERY]
            response_processors = data_processors[RESPONSE]
            label_processors = data_processors[LABEL]

            # Load text tokenizers after loading data processors.
            if TEXT in query_processors:
                query_processors[TEXT] = load_text_tokenizers(
                    text_processors=query_processors[TEXT],
                    path=path,
                )
            # backward compatibility. Add feature column names in each data processor.
            query_processors = assign_feature_column_names(
                data_processors=query_processors,
                df_preprocessor=query_df_preprocessor,
            )
            # Only keep the modalities with non-empty processors.
            query_processors = {k: v for k, v in query_processors.items() if len(v) > 0}

            # Load text tokenizers after loading data processors.
            if TEXT in response_processors:
                response_processors[TEXT] = load_text_tokenizers(
                    text_processors=response_processors[TEXT],
                    path=path,
                )
            # backward compatibility. Add feature column names in each data processor.
            response_processors = assign_feature_column_names(
                data_processors=response_processors,
                df_preprocessor=response_df_preprocessor,
            )
            # Only keep the modalities with non-empty processors.
            response_processors = {k: v for k, v in response_processors.items() if len(v) > 0}
        except:  # backward compatibility. reconstruct the data processor in case something went wrong.
            query_processors = None
            response_processors = None
            label_processors = None

        matcher._query = assets["query"]
        matcher._response = assets["response"]
        matcher._match_label = assets["match_label"]
        matcher._label_column = assets["label_column"]
        matcher._problem_type = assets["problem_type"]
        matcher._eval_metric_name = assets["eval_metric_name"]
        matcher._verbosity = verbosity
        matcher._resume = resume
        matcher._save_path = path  # in case the original exp dir is copied to somewhere else
        matcher._pretrain_path = path
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
        path = os.path.abspath(os.path.expanduser(path))
        assert os.path.isdir(path), f"'{path}' must be an existing directory."
        matcher = cls(query="", response="")
        matcher = cls._load_metadata(matcher=matcher, path=path, resume=resume, verbosity=verbosity)

        query_model, response_model = create_siamese_model(
            query_config=matcher._query_config,
            response_config=matcher._response_config,
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

        query_model, response_model = cls._load_state_dict(
            query_model=query_model,
            response_model=response_model,
            path=load_path,
        )

        matcher._ckpt_path = ckpt_path
        matcher._query_model = query_model
        matcher._response_model = response_model

        if not resume:
            matcher._continuous_training = True

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
