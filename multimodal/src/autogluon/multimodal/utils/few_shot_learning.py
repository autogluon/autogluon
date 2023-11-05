import json
import logging
import os
import pickle
import time
import warnings
from datetime import timedelta
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..constants import BINARY, FEATURE_EXTRACTION, Y_PRED, Y_TRUE
from ..data.infer_types import infer_problem_type
from ..learners import BaseLearner
from ..utils import CustomUnpickler, compute_score, logits_to_prob, setup_save_path

logger = logging.getLogger(__name__)


class FewShotSVMPredictor:
    def __init__(
        self,
        label: Optional[str] = None,
        hyperparameters: Optional[dict] = None,
        presets: Optional[str] = None,
        eval_metric: Optional[str] = None,
        path: Optional[str] = None,
        problem_type: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        label
            Name of the column that contains the target variable to predict.
        hyperparameters
            This is to override some default configurations.
            example:
                hyperparameters = {
                    "model.hf_text.checkpoint_name": "sentence-transformers/all-mpnet-base-v2",
                    "model.hf_text.pooling_mode": "mean",
                    "env.per_gpu_batch_size": 32,
                    "env.eval_batch_size_ratio": 4,
                }
        presets
            Presets regarding model quality, e.g., best_quality, high_quality, and medium_quality.
        eval_metric
            Evaluation metric name.
        path
            Path to directory where models and intermediate outputs should be saved.
            If unspecified, a time-stamped folder called "AutogluonAutoMM/ag-[TIMESTAMP]"
            will be created in the working directory to store all models.
            Note: To call `fit()` twice and save all results of each fit,
            you must specify different `path` locations or don't specify `path` at all.
            Otherwise files from first `fit()` will be overwritten by second `fit()`.
        problem_type
            The problem type specified by user. Currently the SVM predictor only supports classification types
        """
        self.learner = BaseLearner(
            label=None,
            hyperparameters=hyperparameters,
            problem_type=FEATURE_EXTRACTION,
            eval_metric=eval_metric,
            path=path,
            presets=presets,
        )

        self.clf = make_pipeline(StandardScaler(), SVC(gamma="auto"))
        self._fit_called = False
        self._model_loaded = False
        self._save_path = path

        self._label = label
        self._hyperparameters = hyperparameters
        self._presets = presets
        self._eval_metric = eval_metric
        self._problem_type = problem_type
        self._total_train_time = None

    @property
    def problem_type(self):
        return self._problem_type

    def fit(self, train_data: pd.DataFrame):
        training_start = time.time()
        features = self.extract_embedding(data=train_data)
        self._problem_type = infer_problem_type(
            y_train_data=train_data[self._label],
            provided_problem_type=self._problem_type,
        )
        labels = np.array(train_data[self._label])
        self.clf.fit(features, labels)
        self._fit_called = True
        training_end = time.time()
        self._total_train_time = training_end - training_start
        # Automatically save the model after .fit()
        self.save()

    def predict(self, data, as_pandas: Optional[bool] = False, realtime: Optional[bool] = False):
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

        Returns
        -------
        Array of predictions, one corresponding to each row in given dataset.
        """
        if not self._fit_called and not self._model_loaded:
            raise RuntimeError(
                "Neither .fit() nor .load() is not invoked. Please consider calling .fit() or .load() before .predict()"
            )
        features = self.extract_embedding(data, realtime=realtime)

        preds = self.clf.predict(features)
        if as_pandas:
            preds = self.learner._as_pandas(data=data, to_be_converted=preds)
        return preds

    def predict_proba(
        self,
        data,
        as_pandas: Optional[bool] = False,
        as_multiclass: Optional[bool] = True,
        realtime: Optional[bool] = None,
    ):
        """
        Predict probabilities class probabilities rather than class labels.
        This is only for the classification tasks. Calling it for a regression task will throw an exception.

        Parameters
        ----------
        data
            The data to make predictions for. Should contain same column names as training data and
              follow same format (except for the `label` column).
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
        if not self._fit_called and not self._model_loaded:
            warnings.warn(
                "Neither .fit() nor .load() is not invoked. Unexpected predictions may occur. Please consider calling .fit() or .load() before .predict()"
            )

        features = self.extract_embedding(data, realtime=realtime)
        preds = self.clf.decision_function(features)
        probs = logits_to_prob(preds)
        # probs = np.exp(preds) / np.sum(np.exp(preds + 1e-8), axis=1, keepdims=True)

        if not as_multiclass:
            if self._problem_type == BINARY:
                prob = probs[:, 1]

        if (as_pandas is None and isinstance(data, pd.DataFrame)) or as_pandas is True:
            probs = self.learner._as_pandas(data=data, to_be_converted=probs)
        return probs

    def extract_embedding(
        self,
        data: pd.DataFrame,
        realtime: Optional[bool] = False,
    ):
        """
        Extract features for each sample, i.e., one row in the provided dataframe `data`.

        Parameters
        ----------
        data
            The data to extract embeddings for. Should contain same column names as training dataset and
            follow same format (except for the `label` column).
        as_tensor
            Whether to return a Pytorch tensor.
        as_pandas
            Whether to return the output as a pandas DataFrame (True) or numpy array (False).
        realtime
            Whether to do realtime inference, which is efficient for small data (default None).
            If not specified, we would infer it on based on the data modalities
            and sample number.

        Returns
        -------
        Array of embeddings, corresponding to each row in the given data.
        It will have shape (#samples, D) where the embedding dimension D is determined
        by the neural network's architecture.
        """
        if self._label in data.columns:
            data = data.drop(columns=[self._label], axis=1)
        features = self.learner.extract_embedding(data, realtime=realtime)
        assert len(features.keys()) == 1, "Currently SVM only supports single column feature input"
        features_key = list(features.keys())[0]
        features = features[features_key]
        return features

    def evaluate(
        self,
        data: pd.DataFrame,
        metrics: Optional[Union[str, List[str]]] = None,
        return_pred: Optional[bool] = False,
        realtime: Optional[bool] = None,
    ):
        if not self._fit_called and not self._model_loaded:
            warnings.warn(
                "Neither .fit() nor .load() is not invoked. Unexpected predictions may occur. Please consider calling .fit() or .load() before .evaluate()"
            )
        assert (
            self._label in data.columns
        ), f"Label {self._label} is not in the data. Cannot perform evaluation without ground truth labels."
        features = self.extract_embedding(data)
        preds = self.clf.predict(features)
        labels = np.array(data[self._label])

        metric_data = {}
        # TODO: Support BINARY vs. MULTICLASS
        metric_data.update(
            {
                Y_PRED: preds,
                Y_TRUE: labels,
            }
        )
        if metrics is None:
            metrics = [self.learner._eval_metric_name]
        if isinstance(metrics, str):
            metrics = [metrics]

        results = {}
        for per_metric in metrics:
            if per_metric == "macro_f1":
                macro_f1 = f1_score(labels, preds, average="macro")
                results[per_metric] = macro_f1
            else:
                score = compute_score(
                    metric_data=metric_data,
                    metric=per_metric.lower(),
                )
                results[per_metric] = score

        if return_pred:
            return results, self.learner._as_pandas(data=data, to_be_converted=preds)
        return results

    def save_svm(self, path: Optional[str] = None):
        params = self.clf.get_params()
        logger.info(f"Saving into {os.path.join(path, 'svm_model.pkl')}")
        with open(os.path.join(path, "svm_model.pkl"), "wb") as fp:
            pickle.dump(params, fp)

    def load_svm(self, path: Optional[str] = None):
        logger.info(f"Loading from {os.path.join(path, 'svm_model.pkl')}")
        with open(os.path.join(path, "svm_model.pkl"), "rb") as fp:
            params = CustomUnpickler(fp).load()
        self.clf.set_params(**params)
        self._model_loaded = True

    def save_meta_data(self, path: str):
        with open(os.path.join(path, "fewshot_svm_assets.json"), "w") as fp:
            json.dump(
                {
                    "path": self._save_path,
                    "label": self._label,
                    "hyperparameters": self._hyperparameters,
                    "presets": self._presets,
                    "eval_metric": self._eval_metric,
                    "problem_type": self._problem_type,
                },
                fp,
            )

    def load_predictor_from_meta_data(self, path: str):
        with open(os.path.join(path, "fewshot_svm_assets.json"), "r") as fp:
            assets = json.load(fp)
            predictor = BaseLearner(
                label=None,
                hyperparameters=assets["hyperparameters"],
                problem_type=FEATURE_EXTRACTION,
                eval_metric=assets["eval_metric"],
                path=assets["path"],
                presets=assets["presets"],
            )
        return predictor, assets

    @classmethod
    def load(cls, path: str):
        predictor = cls("dummy_label")
        predictor.learner, assets = predictor.load_predictor_from_meta_data(path)
        predictor._save_path = assets["path"]
        predictor._label = assets["label"]
        predictor._hyperparameters = assets["hyperparameters"]
        predictor._eval_metric = assets["eval_metric"]
        predictor._presets = assets["presets"]
        predictor._problem_type = assets["problem_type"]
        predictor.load_svm(path)
        return predictor

    def save(self, path: Optional[str] = None):
        self._save_path = setup_save_path(
            old_save_path=self._save_path,
            proposed_save_path=path,
            raise_if_exist=True,
            warn_if_exist=True,
            fit_called=self._fit_called,
        )
        self.save_meta_data(self._save_path)
        self.save_svm(self._save_path)

    def fit_summary(self, verbosity=0, show_plot=False):
        if self._total_train_time is None:
            logging.info("There is no `best_score` or `total_train_time`. Have you called `predictor.fit()`?")
        else:
            logging.info(
                f"Here's the model summary:"
                f""
                f"The total training time is {timedelta(seconds=self._total_train_time)}"
            )
        results = {
            "training_time": self._total_train_time,
        }
        return results
