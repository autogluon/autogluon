import warnings
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..constants import COLUMN_FEATURES, Y_PRED, Y_TRUE
from ..utils import (
    compute_score,
    extract_from_output,
    predict,
    try_to_infer_pos_label,
    turn_on_off_feature_column_info,
)
from .base import BaseLearner


class FusionSVMClassificationLearner(BaseLearner):
    def __init__(
        self,
        predictor,
    ):
        super().__init__()
        self.clf = make_pipeline(StandardScaler(), SVC(gamma="auto"))

        self.predictor = predictor
        self._fit_called = False

    def fit(self, train_data: Union[pd.DataFrame, dict, list]):
        """
        fit the svm clf with training data
        Parameters
        ----------
        train_data: pd.DataFrame containing training data as in Predictor, or str file containing annotations
        """
        features = self.extract_embedding(data=train_data)
        labels = np.array(train_data[self.predictor._label_column])
        self.clf.fit(features, labels)
        self._fit_called = True

    def extract_embedding(
        self,
        data: Union[pd.DataFrame, dict, list],
        realtime: Optional[bool] = None,
    ):
        self.predictor._verify_inference_ready()
        turn_on_off_feature_column_info(
            data_processors=self.predictor._data_processors,
            flag=True,
        )
        outputs = predict(
            predictor=self.predictor,
            data=data,
            requires_label=False,
            realtime=realtime,
        )

        features = extract_from_output(outputs=outputs, ret_type=COLUMN_FEATURES, as_ndarray=True)

        assert len(features.keys()) == 1, "Currently SVM only supports single column feature input"
        features_key = list(features.keys())[0]
        features = features[features_key]
        return features

    def predict(self, data: Union[pd.DataFrame, dict, list], as_pandas: bool = False, realtime: bool = False):
        assert self._fit_called, ".fit() is not invoked. Please consider calling .fit() before .predict()"
        features = self.extract_embedding(data, realtime=realtime)

        preds = self.clf.predict(features)
        if as_pandas:
            preds = self.predictor._as_pandas(data=data, to_be_converted=preds)
        return preds

    def predict_proba(self, data: Union[pd.DataFrame, dict, list], as_pandas: bool = False, realtime: bool = False):
        features = self.extract_embedding(data, realtime=realtime)
        preds = self.clf.predict_proba(features)
        if as_pandas:
            preds = self.predictor._as_pandas(data=data, to_be_converted=preds)
        return preds

    def evaluate(
        self,
        data: Union[pd.DataFrame, dict, list],
        metrics: Optional[Union[str, List[str]]] = None,
        return_pred: Optional[bool] = False,
        realtime: Optional[bool] = None,
    ):
        assert self._fit_called, ".fit() is not invoked. Please consider calling .fit() before .evaluate()"
        features = self.extract_embedding(data)
        preds = self.clf.predict(features)
        labels = np.array(data[self.predictor._label_column])

        metric_data = {}
        # TODO: Support BINARY and MULTICLASS
        metric_data.update(
            {
                Y_PRED: preds,
                Y_TRUE: labels,
            }
        )

        if self.predictor._df_preprocessor is None:
            data, df_preprocessor, data_processors = self.predictor._on_predict_start(
                data=data,
                requires_label=True,
            )
        else:
            df_preprocessor = self.predictor._df_preprocessor

        if metrics is None:
            metrics = [self.predictor._eval_metric_name]
        if isinstance(metrics, str):
            metrics = [metrics]

        results = {}
        for per_metric in metrics:
            pos_label = try_to_infer_pos_label(
                data_config=self.predictor._config.data,
                label_encoder=df_preprocessor.label_generator,
                problem_type=self.predictor._problem_type,
            )
            score = compute_score(
                metric_data=metric_data,
                metric_name=per_metric.lower(),
                pos_label=pos_label,
            )
            results[per_metric] = score

        if return_pred:
            return results, self.predictor._as_pandas(data=data, to_be_converted=preds)
        return results

    def save(self):
        if not self._fit_called:
            warnings.warn(".fit() is not called. Only saving initial parameters!")
        params = self.clf.get_params()
        pass

    def load(self):
        pass

    def macro_f1_metric(self, metric_data: dict):
        preds = metric_data[Y_PRED]
        labels = metric_data[Y_TRUE]
