import warnings

from autogluon.multimodal.predictor import MultiModalPredictor
from typing import List, Optional, Union

import logging
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..constants import AUTOMM, FEATURE_EXTRACTION, Y_PRED, Y_TRUE

from ..utils import compute_score, try_to_infer_pos_label, setup_save_path

logger = logging.getLogger(AUTOMM)


class FewShotSVMPredictor:
    def __init__(
        self,
        label: Optional[str] = None,
        hyperparameters: Optional[dict] = None,
        presets: Optional[str] = None,
        eval_metric: Optional[str] = None,
        path: Optional[str] = None,
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

        """
        self._automm_predictor = MultiModalPredictor(
            label=label,
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
        pass

    def fit(self, train_data: pd.DataFrame):
        features = self.extract_embedding(data=train_data)
        labels = np.array(train_data[self._automm_predictor._label_column])
        self.clf.fit(features, labels)
        self._fit_called = True
        # Automatically save the model after .fit()
        self.save()

    def predict(self, data, as_pandas: Optional[bool] = False):
        if not self._fit_called and not self._model_loaded:
            warnings.warn(
                "Neither .fit() nor .load() is not invoked. Unexpected predictions may occur. Please consider calling .fit() or .load() before .predict()"
            )
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
        if not self._fit_called and not self._model_loaded:
            warnings.warn(
                "Neither .fit() nor .load() is not invoked. Unexpected predictions may occur. Please consider calling .fit() or .load() before .evaluate()"
            )
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

    def save(self, path: Optional[str] = None):
        self._save_path = setup_save_path(
            old_save_path=self._save_path,
            proposed_save_path=path,
            raise_if_exist=True,
            warn_if_exist=False,
            fit_called=self._fit_called,
        )
        params = self.clf.get_params()
        logger.info(f"Saving into {os.path.join(self._save_path, 'svm_model.pkl')}")
        pickle.dump(params, open(os.path.join(self._save_path, "svm_model.pkl"), "wb"))
        return self._save_path

    def load(self, path: Optional[str] = None):
        if path is not None:
            logger.info(f"Loading from {os.path.join(path, 'svm_model.pkl')}")
            params = pickle.load(open(os.path.join(path, "svm_model.pkl"), "rb"))
        else:
            logger.info(f"Loading from {os.path.join(self._save_path, 'svm_model.pkl')}")
            assert (
                self._save_path is not None
            ), "Expect at least one of path and self._save_path to be not None."
            params = pickle.load(open(os.path.join(self._save_path, "svm_model.pkl"), "rb"))
        self.clf.set_params(**params)
        self._model_loaded = True
