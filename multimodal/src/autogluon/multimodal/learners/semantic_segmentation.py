import logging
import os
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image

from ..constants import LABEL, LOGITS
from ..optimization.utils import get_metric, get_norm_layer_param_names, get_trainable_params_efficient_finetune
from ..utils import extract_from_output, setup_save_path
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
        realtime: Optional[bool] = None,
        save_results: Optional[bool] = None,
        **kwargs,
    ):
        """
        Predict values for the label column of new data.

        Parameters
        ----------
        data
            The data to make predictions for. Should contain same column names as training data and
            follow same format (except for the `label` column).
        as_pandas
            Whether to return the output as a pandas DataFrame(Series) (True) or numpy array (False).
        realtime
            Whether to do realtime inference, which is efficient for small data (default None).
            If not specified, we would infer it on based on the data modalities
            and sample number.
        save_results
            Whether to save the prediction results (only works for detection now)

        Returns
        -------
        Array of predictions, one corresponding to each row in given dataset.
        """
        self.ensure_predict_ready()
        ret_type = LOGITS

        outputs = self.predict_per_run(
            data=data,
            requires_label=False,
            realtime=realtime,
        )
        logits = extract_from_output(outputs=outputs, ret_type=ret_type)

        if self._df_preprocessor:  # 有和没有先fit后predict？结果一样么
            pred = self._df_preprocessor.transform_prediction(
                y_pred=logits,
            )
        else:
            if isinstance(logits, (torch.Tensor, np.ndarray)) and logits.ndim == 2:
                pred = logits.argmax(axis=1)
            else:
                pred = logits

        if save_results:
            self._save_path = setup_save_path(
                old_save_path=self._save_path,
                warn_if_exist=False,
            )
            self.save_segmentation_result(
                pred=pred,
                data=data,
                result_path=self._save_path,
            )

        return pred

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
            Whether to do realtime inference, which is efficient for small data (default None).
            If not specified, we would infer it on based on the data modalities
            and sample number.

        Returns
        -------
        Array of predicted class-probabilities, corresponding to each row in the given data.
        When as_multiclass is True, the output will always have shape (#samples, #classes).
        Otherwise, the output will have shape (#samples,)
        """
        self.ensure_predict_ready()

        outputs = self.predict_per_run(
            data=data,
            requires_label=False,
            realtime=realtime,
        )
        logits = extract_from_output(outputs=outputs, ret_type=LOGITS)
        prob = logits  # for binary
        # prob = logits_to_prob(logits)

        # if not as_multiclass:
        #     if self._problem_type == BINARY:
        #         prob = prob[:, 1]

        # if (as_pandas is None and isinstance(data, pd.DataFrame)) or as_pandas is True:
        #     prob = self._as_pandas(data=data, to_be_converted=prob)

        return prob

    def extract_embedding(
        self,
        data: Union[pd.DataFrame, dict, list],
        as_tensor: Optional[bool] = False,
        as_pandas: Optional[bool] = False,
        realtime: Optional[bool] = None,
        **kwargs,
    ):
        raise NotImplementedError("Semantic segmentation doesn't support calling `extract_embedding` yet.")

    def save_segmentation_result(self, pred: Iterable, data: Union[pd.DataFrame, Dict], result_path: str):
        """
        Saving segmentation results in pd.DataFrame format (per image)

        Parameters
        ----------
        pred
            List containing detection results for one image
        data
            pandas data frame or dict containing the image information to be tested
        result_path
            path to save result
        Returns
        -------
        The paths of the segmentation results as pandas DataFrame
        """
        if isinstance(data, dict):
            image_names = data["image"]
        else:
            image_names = data["image"].to_list()
        results = []

        mask_path = os.path.join(result_path, "masks")
        txt_path = os.path.join(result_path, "result.txt")
        os.makedirs(mask_path, exist_ok=True)
        for image_pred, image_name in zip(pred, image_names):
            mask = Image.fromarray(np.squeeze(image_pred, axis=0))
            per_mask_path = os.path.join(mask_path, os.path.basename(image_name))
            mask.save(per_mask_path)
            results.append([image_name, per_mask_path])

        result_df = pd.DataFrame(results, columns=["image", "mask"])
        result_df.to_csv(txt_path, index=False)
        return result_df
