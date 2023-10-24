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
        problem_type: Optional[str] = OBJECT_DETECTION,
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
        pretrained: Optional[bool] = True,
        sample_data_path: Optional[str] = None,
    ):
        assert (
            problem_type == OBJECT_DETECTION
        ), f"Expected problem_type={OBJECT_DETECTION}, but problem_type={problem_type}"

        check_if_packages_installed(problem_type=problem_type)

        super().__init__(
            problem_type=problem_type,
            query=query,
            response=response,
            match_label=match_label,
            pipeline=pipeline,
            presets=presets,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            path=path,
            verbosity=verbosity,
            num_classes=num_classes,
            classes=classes,
            warn_if_exist=warn_if_exist,
            enable_progress_bar=enable_progress_bar,
            pretrained=pretrained,
            sample_data_path=sample_data_path,
        )

        self._label_column = "label"
        if self._sample_data_path is not None:
            self._classes = get_detection_classes(self._sample_data_path)
            self._output_shape = len(self._classes)

    @property
    def classes(self):
        """
        Return the classes of object detection.
        """
        return self._model.model.CLASSES

    def fit_per_run(self):
        val_use_training_mode = (self._problem_type == OBJECT_DETECTION) and (validation_metric_name != MAP)

    def prepare_for_train_tuning_data(
        self,
        train_data: Union[pd.DataFrame, str],
        tuning_data: Optional[Union[pd.DataFrame, str]],
        holdout_frac: Optional[float],
        max_num_tuning_data: Optional[int],
        seed: Optional[int],
    ):
        # TODO: remove self from calling setup_detection_train_tuning_data()
        train_data, tuning_data = setup_detection_train_tuning_data(
            predictor=self,
            train_data=train_data,
            tuning_data=tuning_data,
            max_num_tuning_data=max_num_tuning_data,
            seed=seed,
        )

        if tuning_data is None:
            train_data, tuning_data = self._split_train_tuning(
                data=train_data, holdout_frac=holdout_frac, random_state=seed
            )

        self._train_data = train_data
        self._tuning_data = tuning_data

    def infer_output_shape(self, **kwargs):
        # TODO: support inferring output during fit()?
        assert self._output_shape is not None, f"output_shape should have been set in the learner initialization."

    def get_datamodule_per_run(self, df_preprocessor, data_processors, config):
        if self._model.config is not None and MULTI_IMAGE_MIX_DATASET in self._model.config:
            train_dataset = MultiImageMixDataset(
                data=self._train_data,
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
                validate_data=self._tuning_data,
                val_use_training_mode=val_use_training_mode,
            )
        else:
            train_dm = BaseDataModule(
                df_preprocessor=df_preprocessor,
                data_processors=data_processors,
                per_gpu_batch_size=self._config.env.per_gpu_batch_size,
                num_workers=self._config.env.num_workers,
                train_data=self._train_data,
                validate_data=self._tuning_data,
                val_use_training_mode=val_use_training_mode,
            )

        return train_dm

    def build_task_per_run(
        self,
        model: Optional[nn.Module] = None,
        optimization_kwargs: Optional[dict] = None,
        is_train=True,
    ):
        if is_train:
            return MMDetLitModule(
                model=model,
                **optimization_kwargs,
            )
        else:
            return MMDetLitModule(model=self._model)

    def on_predict_end(self, pred_writer, outputs):
        if pred_writer is None:
            # TODO: remove this by adjusting the return of mmdet_image or lit_mmdet.
            outputs = [output for batch_outputs in outputs for output in batch_outputs]
        return outputs

    def _on_predict_start(
            self,
            data: Union[pd.DataFrame, dict, list],
            requires_label: bool,
    ):
        data = self.data_to_df(data=data)
        column_types = self.infer_column_types(column_types=self._column_types, data=data, is_train=False)
        column_types = infer_rois_column_type(
            column_types=column_types,
            data=data,
        )
        df_preprocessor = self.get_df_preprocessor_per_run(df_preprocessor=self._df_preprocessor, data=data,
                                                           column_types=column_types, is_train=False)
        if self._fit_called:
            df_preprocessor._column_types = self.update_image_column_types(data=data)
        data_processors = self.get_data_processors_per_run(data_processors=self._data_processors,
                                                           requires_label=requires_label, is_train=False)

        return data, df_preprocessor, data_processors

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