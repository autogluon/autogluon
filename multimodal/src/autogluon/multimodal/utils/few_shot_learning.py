from autogluon.multimodal.predictor import MultiModalPredictor
from typing import List, Optional, Union

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..constants import FEATURE_EXTRACTION, Y_PRED, Y_TRUE

from ..utils import (
    compute_score,
    try_to_infer_pos_label,
)


class FewShotSVMPredictor:
    def __init__(
        self,
        label: Optional[str] = None,
        hyperparameters: dict = None,
        problem_type: str = None,
        eval_metric: str = None,
    ):
        assert (
            problem_type == FEATURE_EXTRACTION
        ), f"Expect the problem type to be {FEATURE_EXTRACTION}, but got {problem_type}"
        self._automm_predictor = MultiModalPredictor(
            label=label, hyperparameters=hyperparameters, problem_type=problem_type, eval_metric=eval_metric
        )

        self.clf = make_pipeline(StandardScaler(), SVC(gamma="auto"))
        self._fit_called = False
        pass

    def fit(self, train_data: pd.DataFrame):
        features = self.extract_embedding(data=train_data)
        labels = np.array(train_data[self._automm_predictor._label_column])
        self.clf.fit(features, labels)
        self._fit_called = True

    def predict(self, data, as_pandas: Optional[bool] = False):
        assert self._fit_called, ".fit() is not invoked. Please consider calling .fit() before .predict()"
        features = self.extract_embedding(data)

        preds = self.clf.predict(features)
        if as_pandas:
            preds = self._automm_predictor._as_pandas(data=data, to_be_converted=preds)
        return preds

    def extract_embedding(self, data: pd.DataFrame):
        features = self._automm_predictor.extract_embedding(data)
        assert len(features.keys()) == 1, "Currently SVM only supports single column feature input"
        features_key = list(features.keys())[0]
        features = features[features_key]
        return features

    def evaluate(
        self, data: pd.DataFrame, metrics: Optional[Union[str, List[str]]] = None, return_pred: Optional[bool] = False
    ):
        assert self._fit_called, ".fit() is not invoked. Please consider calling .fit() before .evaluate()"
        features = self.extract_embedding(data)
        preds = self.clf.predict(features)
        labels = np.array(data[self._automm_predictor._label_column])

        metric_data = {}
        # TODO: Support BINARY vs. MULTICLASS
        metric_data.update(
            {
                Y_PRED: preds,
                Y_TRUE: labels,
            }
        )

        if self._automm_predictor._df_preprocessor is None:
            data, df_preprocessor, data_processors = self._automm_predictor._on_predict_start(
                data=data,
                requires_label=True,
            )
        else:
            df_preprocessor = self._automm_predictor._df_preprocessor

        if metrics is None:
            metrics = [self._automm_predictor._eval_metric_name]
        if isinstance(metrics, str):
            metrics = [metrics]

        results = {}
        for per_metric in metrics:
            if per_metric == "macro_f1":
                macro_f1 = f1_score(labels, preds, average="macro")
                results[per_metric] = macro_f1
            else:
                pos_label = try_to_infer_pos_label(
                    data_config=self._automm_predictor._config.data,
                    label_encoder=df_preprocessor.label_generator,
                    problem_type=self._automm_predictor._problem_type,
                )
                score = compute_score(
                    metric_data=metric_data,
                    metric_name=per_metric.lower(),
                    pos_label=pos_label,
                )
                results[per_metric] = score

        if return_pred:
            return results, self._automm_predictor._as_pandas(data=data, to_be_converted=preds)
        return results

    def save(self):
        pass

    def load(self):
        pass
