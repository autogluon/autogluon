import json
import logging
import os
import pickle
import time
import warnings
from datetime import timedelta
from typing import List, Optional, Union, Dict

import lightning.pytorch as pl
import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from omegaconf import DictConfig

from ..constants import BINARY, FEATURE_EXTRACTION, Y_PRED, Y_TRUE, FEATURES, COLUMN_FEATURES
from ..data.infer_types import infer_problem_type
from ..data.preprocess_dataframe import MultiModalFeaturePreprocessor
from .base import BaseLearner
from ..utils import CustomUnpickler, compute_score, logits_to_prob, setup_save_path, compute_num_gpus, infer_precision, extract_from_output, turn_on_off_feature_column_info

logger = logging.getLogger(__name__)


class FewShotSVMLearner(BaseLearner):
    def __init__(
        self,
        label: Optional[str] = None,
        problem_type: Optional[str] = None,
        hyperparameters: Optional[dict] = None,
        presets: Optional[str] = None,
        eval_metric: Optional[str] = None,
        path: Optional[str] = None,
        verbosity: Optional[int] = 2,
        warn_if_exist: Optional[bool] = True,
        enable_progress_bar: Optional[bool] = None,
        **kwargs,
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
        super().__init__(
            label=label,
            problem_type=problem_type,
            presets=presets,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            path=path,
            verbosity=verbosity,
            warn_if_exist=warn_if_exist,
            enable_progress_bar=enable_progress_bar,
        )

        self._svm = None
        self._model_loaded = False

    def update_attributes(
        self,
        config: Optional[Dict] = None,
        df_preprocessor: Optional[MultiModalFeaturePreprocessor] = None,
        data_processors: Optional[Dict] = None,
        model: Optional[nn.Module] = None,
        svm: Optional[Pipeline] = None,
        **kwargs,
    ):
        super().update_attributes(
            config=config,
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
            model=model,
        )
        if svm:
            self._svm = svm

    def fit_sanity_check(self):
        unique_dtypes = set(self._column_types.values())
        assert len(unique_dtypes) == 1, f"Few shot SVM learner allows single modality data for now, but detected modalities {unique_dtypes}."

    def fit(
        self,
        train_data: Union[pd.DataFrame, str],
        presets: Optional[str] = None,
        config: Optional[Dict] = None,
        tuning_data: Optional[Union[pd.DataFrame, str]] = None,
        time_limit: Optional[int] = None,
        save_path: Optional[str] = None,
        hyperparameters: Optional[Union[str, Dict, List[str]]] = None,
        column_types: Optional[Dict] = None,
        holdout_frac: Optional[float] = None,
        teacher_learner: Union[str, BaseLearner] = None,
        seed: Optional[int] = 0,
        standalone: Optional[bool] = True,
        hyperparameter_tune_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        training_start = self.on_fit_start(presets=presets, config=config)
        self.setup_save_path(save_path=save_path)
        self.prepare_train_tuning_data(
            train_data=train_data,
            tuning_data=tuning_data,
            holdout_frac=holdout_frac,
            seed=seed,
        )
        self.infer_column_types(column_types=column_types)
        self.update_hyperparameters(
            hyperparameters=hyperparameters,
            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
        )
        self.fit_sanity_check()
        self.prepare_fit_args(time_limit=time_limit, seed=seed)
        fit_returns = self.execute_fit()
        self.on_fit_end(training_start=training_start)

        return self

    @staticmethod
    def get_svm_per_run(svm: Pipeline):
        if svm is None:
            svm = make_pipeline(StandardScaler(), SVC(gamma="auto"))
        return svm

    def on_fit_per_run_end(
        self,
        trainer: pl.Trainer,
        config: DictConfig,
        model: nn.Module,
        svm: Pipeline,
        df_preprocessor: MultiModalFeaturePreprocessor,
        data_processors: Dict,
        save_path: str,
    ):
        self.clean_trainer_processes(trainer=trainer, is_train=True)
        self.save(
            path=save_path,
            config=config,
            model=model,
            svm=svm,
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
            fit_called=True,  # fit is called on one run.
            save_model=False,  # The final model will be saved in top_k_average
        )

    def fit_per_run(
        self,
        max_time: timedelta,
        save_path: str,
        enable_progress_bar: bool,
        seed: int,
        hyperparameters: Optional[Union[str, Dict, List[str]]] = None,
        advanced_hyperparameters: Optional[Dict] = None,
        config: Optional[Dict] = None,
        df_preprocessor: Optional[MultiModalFeaturePreprocessor] = None,
        data_processors: Optional[Dict] = None,
        model: Optional[nn.Module] = None,
        svm: Optional[Pipeline] = None,
    ):
        self.on_fit_per_run_start(seed=seed, save_path=save_path)
        config = self.get_config_per_run(config=config, hyperparameters=hyperparameters)
        df_preprocessor = self.get_df_preprocessor_per_run(
            df_preprocessor=df_preprocessor,
            config=config,
        )
        config = self.update_config_by_data_per_run(config=config, df_preprocessor=df_preprocessor)
        model = self.get_model_per_run(model=model, config=config, df_preprocessor=df_preprocessor)
        model = self.compile_model_per_run(config=config, model=model)
        data_processors = self.get_data_processors_per_run(
            data_processors=data_processors,
            config=config,
            model=model,
            advanced_hyperparameters=advanced_hyperparameters,
        )
        turn_on_off_feature_column_info(
            data_processors=data_processors,
            flag=True,
        )
        if max_time == timedelta(seconds=0):
            return dict(
                config=config,
                df_preprocessor=df_preprocessor,
                data_processors=data_processors,
                model=model,
            )
        datamodule = self.get_datamodule_per_run(
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
            per_gpu_batch_size=config.env.per_gpu_batch_size,
            num_workers=config.env.num_workers,
        )
        num_gpus = compute_num_gpus(config_num_gpus=config.env.num_gpus, strategy=config.env.strategy)
        self.log_gpu_info(num_gpus=num_gpus, config=config)
        precision = infer_precision(num_gpus=num_gpus, precision=config.env.precision)
        strategy = self.get_strategy_per_run(num_gpus=num_gpus, config=config)
        config = self.post_update_config_per_run(
            config=config,
            num_gpus=num_gpus,
            precision=precision,
            strategy=strategy,
        )
        pred_writer = self.get_pred_writer(strategy=strategy)
        callbacks = self.get_callbacks_per_run(pred_writer=pred_writer, is_train=False)
        litmodule = self.get_litmodule_per_run(model=model)
        trainer = self.init_trainer_per_run(
            num_gpus=num_gpus,
            config=config,
            precision=precision,
            strategy=strategy,
            max_time=max_time,
            callbacks=callbacks,
            enable_progress_bar=enable_progress_bar,
            is_train=False,  # only use a pretrained model to extract embeddings
        )
        outputs = self.run_trainer(
            trainer=trainer,
            litmodule=litmodule,
            datamodule=datamodule,
            pred_writer=pred_writer,
            is_train=False,  # only use a pretrained model to extract embeddings
        )
        outputs = self.collect_predictions(
            outputs=outputs,
            trainer=trainer,
            pred_writer=pred_writer,
            num_gpus=num_gpus,
        )
        self.clean_trainer_processes(trainer=trainer, is_train=False)

        features = extract_from_output(outputs=outputs, ret_type=COLUMN_FEATURES, as_ndarray=True)
        features = self.aggregate_column_features(features=features)
        labels = np.array(self._train_data[self._label_column])
        svm = self.get_svm_per_run(svm=svm)
        svm.fit(features, labels)

        self.on_fit_per_run_end(
            save_path=save_path,
            trainer=trainer,
            config=config,
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
            model=model,
            svm=svm,
        )

        return dict(
            config=config,
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
            model=model,
            svm=svm,
        )

    def aggregate_column_features(self, features: Union[Dict, torch.Tensor]):
        if isinstance(features, torch.Tensor):
            return features
        elif isinstance(features, dict):
            assert len(features) != 0, f"column features are empty."
            if len(features) == 1:
                return next(iter(features.values()))
            if self._config.data.aggregate_column_features == "concat":
                return torch.cat(features.values(), dim=1)
            elif self._config.data.aggregate_column_features == "mean":
                return torch.mean(torch.stack(features.values()), dim=0)
            else:
                raise ValueError(f"Unsupported aggregate_column_features value {self._config.data.aggregate_column_features}.")
        else:
            raise ValueError(f"Unsupported features type: {type(features)} in aggregating column features.")

    def predict(
        self,
        data: Union[pd.DataFrame, dict, list, str],
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
        self.on_predict_start()
        features = self.extract_embedding(data=data, realtime=realtime, as_tensor=True)
        features = self.aggregate_column_features(features=features)
        pred = self._svm.predict(features)
        if (as_pandas is None and isinstance(data, pd.DataFrame)) or as_pandas is True:
            pred = self._as_pandas(data=data, to_be_converted=pred)
        return pred

    def predict_proba(
        self,
        data,
        as_pandas: Optional[bool] = False,
        as_multiclass: Optional[bool] = True,
        realtime: Optional[bool] = None,
        **kwargs,
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
        self.on_predict_start()
        features = self.extract_embedding(data, realtime=realtime, as_tensor=True)
        features = self.aggregate_column_features(features=features)
        logits = self._svm.decision_function(features)
        prob = logits_to_prob(logits)
        if not as_multiclass:
            if self._problem_type == BINARY:
                prob = prob[:, 1]
        if (as_pandas is None and isinstance(data, pd.DataFrame)) or as_pandas is True:
            prob = self._as_pandas(data=data, to_be_converted=prob)
        return prob

    def extract_embedding(
        self,
        data: pd.DataFrame,
        realtime: Optional[bool] = False,
        as_tensor: Optional[bool] = False,
        as_pandas: Optional[bool] = False,
        **kwargs,
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
        self.on_predict_start()
        turn_on_off_feature_column_info(
            data_processors=self._data_processors,
            flag=True,
        )
        outputs = self.predict_per_run(
            data=data,
            requires_label=False,
            realtime=realtime,
        )
        features = extract_from_output(outputs=outputs, ret_type=COLUMN_FEATURES, as_ndarray=as_tensor is False)
        if len(features) == 1:
            return next(iter(features.values()))
        else:
            return features

    def evaluate(
        self,
        data: pd.DataFrame,
        metrics: Optional[Union[str, List[str]]] = None,
        return_pred: Optional[bool] = False,
        realtime: Optional[bool] = None,
    ):
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

    def load_svm(self, path: Optional[str] = None):
        logger.info(f"Loading from {os.path.join(path, 'svm_model.pkl')}")
        with open(os.path.join(path, "svm_model.pkl"), "rb") as fp:
            params = CustomUnpickler(fp).load()
        self.clf.set_params(**params)
        self._model_loaded = True


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

    def save(
        self,
        path: str,
        standalone: Optional[bool] = True,
        config: Optional[DictConfig] = None,
        model: Optional[nn.Module] = None,
        df_preprocessor: Optional[MultiModalFeaturePreprocessor] = None,
        data_processors: Optional[Dict] = None,
        svm: Optional[Pipeline] = None,
        fit_called: Optional[bool] = None,
        save_model: Optional[bool] = True,
    ):
        super().save(
            path=path,
            standalone=standalone,
            config=config,
            model=model,
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
            fit_called=fit_called,
            save_model=save_model,
        )
        svm = svm if svm else self._svm
        with open(os.path.join(path, "svm.pkl"), "wb") as fp:
            pickle.dump(svm.get_params(), fp)

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
