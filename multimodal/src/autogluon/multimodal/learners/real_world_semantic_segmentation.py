import logging
import os
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from ..constants import LABEL, LOGITS
from ..optimization.utils import get_metric, get_norm_layer_param_names, get_trainable_params_efficient_finetune
from ..utils import extract_from_output
from .base import BaseLearner

logger = logging.getLogger(__name__)


class SemanticSegmentationLearner(BaseLearner):
    @staticmethod
    def get_peft_param_names_per_run(model, config):
        peft_param_names = None
        peft = OmegaConf.select(config, "optimization.efficient_finetune")
        if peft:
            norm_param_names = get_norm_layer_param_names(model)
            peft_param_names = get_trainable_params_efficient_finetune(
                norm_param_names,
                efficient_finetune=peft,
                extra_params=OmegaConf.select(config, "optimization.extra_trainable_params"),
            )
        return peft_param_names

    def evaluate_semantic_segmentation(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        metrics: Optional[Union[str, List[str]]] = None,
        return_pred: Optional[bool] = False,
        realtime: Optional[bool] = None,
    ):
        """
        Evaluate semantic segmentation on a test dataset based on "torchmetrics".

        Parameters
        ----------
        data
            A dataframe, containing the same columns as the training data.
            Or a str, that is a path of the annotation file for detection.
        metrics
            Metrics used for evaluation.
        return_pred
            Whether to return the prediction result of each row.
        realtime
            Whether to do realtime inference, which is efficient for small data (default None).
            If not specified, we would infer it on based on the data modalities
            and sample number.
        """
        outputs = self.predict_per_run(
            data=data,
            requires_label=True,
            realtime=realtime,
        )

        logits = extract_from_output(ret_type=LOGITS, outputs=outputs, as_ndarray=False)
        y_pred = logits.float()
        y_true = [ele[LABEL] for ele in outputs]
        y_true = torch.cat(y_true)

        assert len(y_true) == len(y_pred)

        results = {}
        for per_metric_name in metrics:
            per_metric, _ = get_metric(metric_name=per_metric_name.lower())
            per_metric.update(y_pred, y_true)
            score = per_metric.compute()

            results[per_metric_name] = score.item()

        if return_pred:
            return results, outputs
        else:
            return results

    def evaluate(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        metrics: Optional[Union[str, List[str]]] = None,
        return_pred: Optional[bool] = False,
        realtime: Optional[bool] = None,
        **kwargs,
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
        self.ensure_predict_ready()
        return self.evaluate_semantic_segmentation(data, metrics, realtime)

    def predict(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        candidate_data: Optional[Union[pd.DataFrame, dict, list]] = None,
        as_pandas: Optional[bool] = None,
        realtime: Optional[bool] = None,
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
            Whether to do realtime inference, which is efficient for small data (default None).
            If not specified, we would infer it on based on the data modalities
            and sample number.

        Returns
        -------
        Array of predictions, one corresponding to each row in given dataset.
        """
        self.ensure_predict_ready()
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
                requires_label=False,
                realtime=realtime,
            )
            logits = extract_from_output(outputs=outputs, ret_type=ret_type, as_ndarray=False)

            if self._df_preprocessor:
                pred = self._df_preprocessor.transform_prediction(
                    y_pred=logits,
                )
            else:
                if isinstance(logits, (torch.Tensor, np.ndarray)) and logits.ndim == 2:
                    pred = logits.argmax(axis=1)
                else:
                    pred = logits

        # if (as_pandas is None and isinstance(data, pd.DataFrame)) or as_pandas is True:
        #     # TODO
        #     pred = self._as_pandas(data=data, to_be_converted=pred)

        return pred

    def predict_proba(
        self,
        data: Union[pd.DataFrame, dict, list],
        as_pandas: Optional[bool] = None,
        as_multiclass: Optional[bool] = True,
        realtime: Optional[bool] = None,
        **kwargs,
    ):
        raise NotImplementedError("Semantic segmentation doesn't support calling `predict_proba` yet.")

    def extract_embedding(
        self,
        data: Union[pd.DataFrame, dict, list],
        as_tensor: Optional[bool] = False,
        as_pandas: Optional[bool] = False,
        realtime: Optional[bool] = None,
        **kwargs,
    ):
        raise NotImplementedError("Semantic segmentation doesn't support calling `extract_embedding` yet.")
