from __future__ import annotations

import os
import warnings
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
import json

from .predictor import BaseMultiModalPredictor
from .matcher import BaseMultiModalMatcher
from .presets import predictor_presets, matcher_presets, automm_presets

predictor_cls_name = "BaseMultiModalPredictor"
matcher_cls_name = "BaseMultiModalMatcher"


class MultiModalPredictor:
    def __init__(
            self,
            pipeline: Optional[str] = None,
            label: Optional[str] = None,
            problem_type: Optional[str] = None,
            query: Optional[Union[str, List[str]]] = None,
            response: Optional[Union[str, List[str]]] = None,
            negative: Optional[Union[str, List[str]]] = None,
            match_label: Optional[Union[int, str]] = None,
            val_metric: Optional[str] = None,
            eval_metric: Optional[str] = None,
            hyperparameters: Optional[dict] = None,
            path: Optional[str] = None,
            verbosity: Optional[int] = 3,
            num_classes: Optional[int] = None,  # TODO: can we infer this from data?
            classes: Optional[list] = None,
            warn_if_exist: Optional[bool] = True,
            enable_progress_bar: Optional[bool] = None,
            init_scratch: Optional[bool] = False,
    ):
        if pipeline in predictor_presets.list_keys() or pipeline is None:
            print("init predictor...")
            self._predictor = BaseMultiModalPredictor(
                label=label,
                problem_type=problem_type,
                pipeline=pipeline,
                val_metric=val_metric,
                eval_metric=eval_metric,
                hyperparameters=hyperparameters,
                num_classes=num_classes,
                classes=classes,
                path=path,
                verbosity=verbosity,
                warn_if_exist=warn_if_exist,
                enable_progress_bar=enable_progress_bar,
                init_scratch=init_scratch,
            )
            self._class_name = self._predictor.__class__.__name__

        elif pipeline in matcher_presets.list_keys():
            print("init matcher...")
            self._predictor = BaseMultiModalMatcher(
                query=query,
                response=response,
                negative=negative,
                label=label,
                match_label=match_label,
                problem_type=problem_type,
                pipeline=pipeline,
                hyperparameters=hyperparameters,
                eval_metric=eval_metric,
                path=path,
                verbosity=verbosity,
                warn_if_exist=warn_if_exist,
                enable_progress_bar=enable_progress_bar,
            )
            self._class_name = matcher_cls_name

        else:
            raise ValueError(f"Unknown pipeline: {pipeline}. The current supported pipelines are: {automm_presets.list_keys()}")

        self.verbosity = verbosity

    def set_verbosity(self, verbosity: int):
        self._predictor.set_verbosity(verbosity=verbosity)

    @property
    def path(self):
        return self._predictor.path

    @property
    def label(self):
        return self._predictor.label

    @property
    def query(self):
        if self._class_name == matcher_cls_name:
            return self._predictor.query
        else:
            warnings.warn("No query columns are available.", UserWarning)
            return None

    @property
    def response(self):
        if self._class_name == matcher_cls_name:
            return self._predictor.response
        else:
            warnings.warn("No response columns are available.", UserWarning)
            return None

    @property
    def match_label(self):
        if self._class_name == matcher_cls_name:
            return self._predictor.response
        else:
            warnings.warn("No response columns are available.", UserWarning)
            return None

    @property
    def problem_type(self):
        return self._predictor.problem_type

    @property
    def column_types(self):
        return self._predictor.column_types

    @property
    def positive_class(self):
        if self._class_name == predictor_cls_name:
            return self._predictor.positive_class
        else:
            warnings.warn("No positive class is available.", UserWarning)
            return None

    @property
    def class_labels(self):
        return self._predictor.class_labels

    @property
    def class_labels_internal(self):
        """The internal integer labels.

        For example, if the possible labels are ["entailment", "contradiction", "neutral"],
        the internal labels can be [0, 1, 2]

        Returns
        -------
        ret
            List that contains the internal integer labels. It will be None if the predictor is not solving a classification problem.
        """
        if self._predictor.class_labels is None:
            return None
        return list(range(len(self.class_labels)))

    @property
    def class_labels_internal_map(self):
        """The map that projects label names to the internal ids. For example,
        if the internal labels are ["entailment", "contradiction", "neutral"] and the
        internal ids are [0, 1, 2], the label mapping will be
        {"entailment": 0, "contradiction": 1, "neutral": 2}

        Returns
        -------
        ret
            The label mapping dictionary. It will be None if the predictor is not solving a classification problem.
        """
        if self._predictor.class_labels is None:
            return None
        return {k: v for k, v in zip(self.class_labels, self.class_labels_internal)}

    def fit(self,
            train_data: Union[pd.DataFrame, str],
            presets: Optional[str] = None,
            config: Optional[dict] = None,
            tuning_data: Optional[Union[pd.DataFrame, str]] = None,
            id_mappings: Optional[Dict[str, Dict]] = None,
            time_limit: Optional[int] = None,
            save_path: Optional[str] = None,
            hyperparameters: Optional[Union[str, Dict, List[str]]] = None,
            column_types: Optional[dict] = None,
            holdout_frac: Optional[float] = None,
            teacher_predictor: Union[str, BaseMultiModalPredictor] = None,
            seed: Optional[int] = 123,
            standalone: Optional[bool] = True,
            hyperparameter_tune_kwargs: Optional[dict] = None,
            ):

        if self._class_name == predictor_cls_name:
            self._predictor.fit(
                train_data=train_data,
                presets=presets,
                config=config,
                tuning_data=tuning_data,
                time_limit=time_limit,
                hyperparameters=hyperparameters,
                column_types=column_types,
                holdout_frac=holdout_frac,
                save_path=save_path,
                teacher_predictor=teacher_predictor,
                standalone=standalone,
                hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                seed=seed,
            )
        elif self._class_name == matcher_cls_name:
            self._predictor.fit(
                train_data=train_data,
                tuning_data=tuning_data,
                id_mappings=id_mappings,
                time_limit=time_limit,
                presets=presets,
                hyperparameters=hyperparameters,
                column_types=column_types,
                holdout_frac=holdout_frac,
                save_path=save_path,
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown predictor class: {self._class_name}.")

        return self

    def evaluate(
            self,
            data: Union[pd.DataFrame, dict, list, str],
            metrics: Optional[Union[str, List[str]]] = None,
            return_pred: Optional[bool] = False,
            realtime: Optional[bool] = None,
            query_data: Optional[list] = None,
            response_data: Optional[list] = None,
            id_mappings: Optional[Dict[str, Dict]] = None,
            chunk_size: Optional[int] = 1024,
            similarity_type: Optional[str] = "cosine",
            top_k: Optional[int] = 100,
            cutoff: Optional[List[int]] = [5, 10, 20],
            label_column: Optional[str] = None,
            eval_tool: Optional[str] = None,
            seed: Optional[int] = 123,
    ):

        if self._class_name == predictor_cls_name:
            return self._predictor.evaluate(
                data=data,
                metrics=metrics,
                return_pred=return_pred,
                realtime=realtime,
                eval_tool=eval_tool,
                seed=seed,
            )
        elif self._class_name == matcher_cls_name:
            return self._predictor.evaluate(
                data=data,
                query_data=query_data,
                response_data=response_data,
                id_mappings=id_mappings,
                chunk_size=chunk_size,
                similarity_type=similarity_type,
                top_k=top_k,
                cutoff=cutoff,
                label_column=label_column,
                metrics=metrics,
            )
        else:
            raise ValueError(f"Unknown predictor class: {self._class_name}.")

    def predict(
            self,
            data: Union[pd.DataFrame, dict, list],
            candidate_data: Optional[Union[pd.DataFrame, dict, list]] = None,
            id_mappings: Optional[Dict[str, Dict]] = None,
            as_pandas: Optional[bool] = None,
            realtime: Optional[bool] = None,
            seed: Optional[int] = 123,
    ):

        if self._class_name == predictor_cls_name:
            return self._predictor.predict(
                data=data,
                candidate_data=candidate_data,
                as_pandas=as_pandas,
                realtime=realtime,
                seed=seed,
            )
        elif self._class_name == matcher_cls_name:
            return self._predictor.predict(
                data=data,
                id_mappings=id_mappings,
                as_pandas=as_pandas,
            )
        else:
            raise ValueError(f"Unknown predictor class: {self._class_name}.")

    def predict_proba(
            self,
            data: Union[pd.DataFrame, dict, list],
            candidate_data: Optional[Union[pd.DataFrame, dict, list]] = None,
            id_mappings: Optional[Dict[str, Dict]] = None,
            as_pandas: Optional[bool] = None,
            as_multiclass: Optional[bool] = True,
            realtime: Optional[bool] = None,
            seed: Optional[int] = 123,
    ):
        if self._class_name == predictor_cls_name:
            return self._predictor.predict_proba(
                data=data,
                candidate_data=candidate_data,
                as_pandas=as_pandas,
                as_multiclass=as_multiclass,
                realtime=realtime,
                seed=seed,
            )
        elif self._class_name == matcher_cls_name:
            return self._predictor.predict_proba(
                data=data,
                id_mappings=id_mappings,
                as_pandas=as_pandas,
                as_multiclass=as_multiclass,
            )
        else:
            raise ValueError(f"Unknown predictor class: {self._class_name}.")

    def extract_embedding(
            self,
            data: Union[pd.DataFrame, dict, list],
            signature: Optional[str] = None,
            id_mappings: Optional[Dict[str, Dict]] = None,
            return_masks: Optional[bool] = False,
            as_tensor: Optional[bool] = False,
            as_pandas: Optional[bool] = False,
            realtime: Optional[bool] = None,
    ):

        if self._class_name == predictor_cls_name:
            return self._predictor.extract_embedding(
                data=data,
                return_masks=return_masks,
                as_tensor=as_tensor,
                as_pandas=as_pandas,
                realtime=realtime,
            )
        elif self._class_name == matcher_cls_name:
            return self._predictor.extract_embedding(
                data=data,
                signature=signature,
                id_mappings=id_mappings,
                as_tensor=as_tensor,
                as_pandas=as_pandas,
            )
        else:
            raise ValueError(f"Unknown predictor class: {self._class_name}.")

    def save(self, path, standalone=True):
        if self._class_name == predictor_cls_name:
            self._predictor.save(path=path, standalone=standalone)
        elif self._class_name == matcher_cls_name:
            self._predictor.save(path=path, standalone=standalone)
        else:
            raise ValueError(f"Unknown predictor class: {self._class_name}.")

    @classmethod
    def load(
            cls,
            path: str,
            verbosity: int = None,
            resume: bool = False,
    ):
        predictor = cls(label="dummy_label")

        with open(os.path.join(path, "assets.json"), "r") as fp:
            assets = json.load(fp)

        if "class_name" not in assets or assets["class_name"] == predictor_cls_name:
            _predictor = BaseMultiModalPredictor.load(
                path=path,
                resume=resume,
                verbosity=verbosity,
            )
            predictor._class_name = predictor_cls_name
        elif assets["class_name"] == matcher_cls_name:
            _predictor = BaseMultiModalPredictor.load(
                path=path,
                resume=resume,
                verbosity=verbosity,
            )
            predictor._class_name = matcher_cls_name
        else:
            raise ValueError(f"Unknown predictor class {assets['class_name']} is detected in `assets.json`.")

        predictor._predictor = _predictor

        return predictor

    def evaluate_coco(
            self,
            anno_file_or_df: str,
            metrics: str,
            return_pred: Optional[bool] = False,
            seed: Optional[int] = 123,
            eval_tool: Optional[str] = None,
    ):
        assert self._class_name == predictor_cls_name, f"predictor_cls_name is {self._class_name}, but it needs to be BaseMultiModalPredictor to call `evaluate_coco()`."

        return self._predictor.evaluate_coco(
            anno_file_or_df=anno_file_or_df,
            metrics=metrics,
            return_pred=return_pred,
            seed=seed,
            eval_tool=eval_tool,
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
        assert self._class_name == predictor_cls_name, f"predictor_cls_name is {self._class_name}, but it needs to be BaseMultiModalPredictor to call `get_processed_batch_for_deployment()`."

        return self._predictor.get_matcher_data_processors(
            data=data,
            valid_input=valid_input,
            onnx_tracing=onnx_tracing,
            batch_size=batch_size,
            to_numpy=to_numpy,
            requires_label=requires_label,
        )

    def export_onnx(
            self,
            onnx_path: Optional[str] = None,
            data: Optional[pd.DataFrame] = None,
            batch_size: Optional[int] = None,
            verbose: Optional[bool] = False,
            opset_version: Optional[int] = 13,
    ):
        assert self._class_name == predictor_cls_name, f"predictor_cls_name is {self._class_name}, but `export_onnx()` is currently only supported for BaseMultiModalPredictor."
        self._predictor.export_onnx(
            onnx_path=onnx_path,
            data=data,
            batch_size=batch_size,
            verbose=verbose,
            opset_version=opset_version,
        )

    def set_num_gpus(self, num_gpus):
        self._predictor.set_num_gpus(num_gpus=num_gpus)
