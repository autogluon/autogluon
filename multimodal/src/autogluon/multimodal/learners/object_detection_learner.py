import os
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from timm.data.mixup import Mixup
from torch import nn

from autogluon.core.utils.loaders import load_pd

from ..constants import BBOX, MULTI_IMAGE_MIX_DATASET, OBJECT_DETECTION, XYWH
from ..data.datamodule import BaseDataModule
from ..data.dataset_mmlab import MultiImageMixDataset
from ..data.infer_types import infer_problem_type_output_shape
from ..optimization.lit_mmdet import MMDetLitModule
from ..utils.object_detection import (
    convert_pred_to_xywh,
    evaluate_coco,
    get_detection_classes,
    object_detection_data_to_df,
    save_result_df,
    setup_detection_train_tuning_data,
)
from ..utils.pipeline import init_pretrained
from ..utils.save import setup_save_path
from .base_learner import BaseLearner


class ObjectDetectionLearner(BaseLearner):
    def __init__(
        self,
        label: Optional[str] = None,
        problem_type: Optional[str] = None,
        query: Optional[Union[str, List[str]]] = None,
        response: Optional[Union[str, List[str]]] = None,
        match_label: Optional[Union[int, str]] = None,
        pipeline: Optional[str] = None,
        presets: Optional[str] = None,
        eval_metric: Optional[str] = None,
        hyperparameters: Optional[dict] = None,
        path: Optional[str] = None,
        verbosity: Optional[int] = 2,
        num_classes: Optional[int] = None,  # TODO: can we infer this from data?
        classes: Optional[list] = None,
        warn_if_exist: Optional[bool] = True,
        enable_progress_bar: Optional[bool] = None,
        init_scratch: Optional[bool] = False,
        sample_data_path: Optional[str] = None,
    ):
        assert (
            problem_type == OBJECT_DETECTION
        ), f"Expected problem_type={OBJECT_DETECTION}, but problem_type={problem_type}"
        super().__init__(
            label,
            problem_type,
            query,
            response,
            match_label,
            pipeline,
            presets,
            eval_metric,
            hyperparameters,
            path,
            verbosity,
            num_classes,
            classes,
            warn_if_exist,
            enable_progress_bar,
            init_scratch,
            sample_data_path,
        )
        self._validation_metric_name = self._config["optimization"][
            "val_metric"
        ]  # TODO: only object detection is using this

    def _setup_train_tuning_data(
        self,
        train_data: Union[pd.DataFrame, str],
        tuning_data: Optional[Union[pd.DataFrame, str]],
        holdout_frac: Optional[float],
        seed: Optional[int],
        max_num_tuning_data: Optional[int],
        **kwargs,
    ):
        train_data, tuning_data = setup_detection_train_tuning_data(
            self, max_num_tuning_data, seed, train_data, tuning_data
        )
        if tuning_data is None:
            train_data, tuning_data = self._split_train_tuning(
                data=train_data, holdout_frac=holdout_frac, random_state=seed
            )

        return train_data, tuning_data

    def _infer_output_shape(self, **kwargs):
        assert self._output_shape is not None, f"self._output_shape should have been set in __init__"

        return self._output_shape

    def _get_data_module(
        self, train_df, val_df, df_preprocessor, data_processors, val_use_training_mode
    ) -> pl.LightningDataModule:
        if self._model.config is not None and MULTI_IMAGE_MIX_DATASET in self._model.config:
            train_dataset = MultiImageMixDataset(
                data=train_df,
                preprocessor=[df_preprocessor],
                processors=[data_processors],
                model_config=self._model.config,
                id_mappings=None,
                is_training=True,
            )
            train_dm = BaseDataModule(
                df_preprocessor=df_preprocessor,
                data_processors=data_processors,
                per_gpu_batch_size=self._config.env.per_gpu_batch_size,
                num_workers=self._config.env.num_workers,
                train_dataset=train_dataset,
                validate_data=val_df,
                val_use_training_mode=val_use_training_mode,
            )
        else:
            train_dm = BaseDataModule(
                df_preprocessor=df_preprocessor,
                data_processors=data_processors,
                per_gpu_batch_size=self._config.env.per_gpu_batch_size,
                num_workers=self._config.env.num_workers,
                train_data=train_df,
                validate_data=val_df,
                val_use_training_mode=val_use_training_mode,
            )

        return train_dm

    def _get_lightning_module(
        self,
        optimization_kwargs: Optional[dict] = None,
        metrics_kwargs: Optional[dict] = None,
        test_time: bool = False,
        **kwargs,
    ):
        if test_time:
            task = MMDetLitModule(
                model=self._model,
                **optimization_kwargs,
            )
        else:
            task = MMDetLitModule(
                model=self._model,
                **metrics_kwargs,
                **optimization_kwargs,
            )
        return task

    def evaluate(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        # query_data: Optional[list] = None,
        # response_data: Optional[list] = None,
        # id_mappings: Optional[Union[Dict[str, Dict], Dict[str, pd.Series]]] = None,
        metrics: Optional[Union[str, List[str]]] = None,
        # chunk_size: Optional[int] = 1024,
        # similarity_type: Optional[str] = "cosine",
        # cutoffs: Optional[List[int]] = [1, 5, 10],
        # label: Optional[str] = None,
        return_pred: Optional[bool] = False,
        realtime: Optional[bool] = None,
        eval_tool: Optional[str] = None,
        **kwargs,
    ):
        self._verify_inference_ready()
        if realtime:
            return NotImplementedError(f"Current problem type {self._problem_type} does not support realtime predict.")
        if isinstance(data, str):
            return evaluate_coco(
                predictor=self,
                anno_file_or_df=data,
                metrics=metrics,
                return_pred=return_pred,
                eval_tool=eval_tool,
            )
        else:
            data = object_detection_data_to_df(data)
            return evaluate_coco(
                predictor=self,
                anno_file_or_df=data,
                metrics=metrics,
                return_pred=return_pred,
                eval_tool="torchmetrics",
            )
