import os
from typing import Dict, List, Optional, Union
import numpy as np

import pandas as pd
import pytorch_lightning as pl
from timm.data.mixup import Mixup
from torch import nn

from ..constants import BBOX, XYWH
from ..data.infer_types import infer_problem_type_output_shape
from ..optimization.lit_mmdet import MMDetLitModule
from ..utils.save import setup_save_path
from ..utils.object_detection import (
    convert_pred_to_xywh,
    evaluate_coco,
    get_detection_classes,
    object_detection_data_to_df,
    setup_detection_train_tuning_data,
    save_result_df,
)
from ..utils.pipeline import init_pretrained

from .default_learner import DefaultLearner


class ObjectDetectionLearner(DefaultLearner):
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
        hyperparameters: Optional[dict] = None,
        path: Optional[str] = None,
        verbosity: Optional[int] = 2,
        num_classes: Optional[int] = None,
        classes: Optional[list] = None,
        enable_progress_bar: Optional[bool] = True,
        init_scratch: Optional[bool] = False,
        sample_data_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            problem_type=problem_type,
            label=label,
            presets=presets,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            path=path,
            verbosity=verbosity,
            num_classes=num_classes,
            classes=classes,
            enable_progress_bar=enable_progress_bar,
            init_scratch=init_scratch,
            **kwargs,
        )
        if sample_data_path is not None:
            self._classes = get_detection_classes(sample_data_path)
            self._output_shape = len(self._classes)
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
    ):
        train_data, tuning_data = setup_detection_train_tuning_data(
            self, max_num_tuning_data, seed, train_data, tuning_data
        )
        return super()._setup_train_tuning_data(train_data, tuning_data, holdout_frac, seed)

    def _infer_output_shape(self, train_data):
        _, output_shape = infer_problem_type_output_shape(
            label_column=self._label_column,
            column_types=self._column_types,
            data=train_data,
            provided_problem_type=self._problem_type,
        )
        return output_shape

    def _setup_train_task_lightning_module(
        self,
        optimization_kwargs: dict,
        metrics_kwargs: dict,
        **kwargs,
    ) -> pl.LightningModule:
        assert (
            self._model is not None
        ), "self._model is None. You must setup self._model before calling _setup_task_lightning_module()"
        task = MMDetLitModule(
            model=self._model,
            **metrics_kwargs,
            **optimization_kwargs,
        )
        return task

    def _setup_train_task_lightning_module(
        self,
        optimization_kwargs: dict,
        **kwargs,
    ) -> pl.LightningModule:
        assert (
            self._model is not None
        ), "self._model is None. You must setup self._model before calling _setup_task_lightning_module()"
        task = MMDetLitModule(
            model=self._model,
            **optimization_kwargs,
        )
        return task

    def _get_prediction_ret_type(self) -> str:
        return BBOX

    def _postprocess_pred(self, data, pred):
        if self._model.output_bbox_format == XYWH:
            pred = convert_pred_to_xywh(pred)
        return pred

    def _transform_data_for_predict(self, data):
        data = object_detection_data_to_df(data)
        if self._label_column not in data:
            self._label_column = None
        return data

    def _save_prediction(self, data, pred):
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

    def _as_pandas(
        self,
        data: Union[pd.DataFrame, dict, list],
        to_be_converted: Union[np.ndarray, dict],
    ):
        pred = save_result_df(
            pred=to_be_converted,
            data=data,
            detection_classes=self._model.model.CLASSES,
            result_path=None,
        )

        return pred

    def evaluate(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        metrics: Optional[Union[str, List[str]]] = None,
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
