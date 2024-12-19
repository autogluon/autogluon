import logging
import os
import pickle
from datetime import timedelta
from typing import Dict, List, Optional, Union

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn

from autogluon.core.metrics import Scorer
from autogluon.core.utils.loaders import load_pd

from ..constants import CLIP, COLUMN_FEATURES, HF_TEXT, TIMM_IMAGE, Y_PRED, Y_TRUE
from ..data import BaseDataModule, MultiModalFeaturePreprocessor, data_to_df, turn_on_off_feature_column_info
from ..models import select_model
from ..optim import compute_score
from ..utils import LogFilter, apply_log_filter, extract_from_output, get_available_devices, logits_to_prob
from .base import BaseLearner

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
                    "env.inference_batch_size_ratio": 4,
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

    def prepare_train_tuning_data(
        self,
        train_data: Union[pd.DataFrame, str],
        tuning_data: Optional[Union[pd.DataFrame, str]],
        holdout_frac: Optional[float],
        seed: Optional[int],
    ):
        if isinstance(train_data, str):
            train_data = load_pd.load(train_data)
        if isinstance(tuning_data, str):
            tuning_data = load_pd.load(tuning_data)

        self._train_data = train_data
        self._tuning_data = tuning_data  # TODO: use tuning_data in few shot learning?

    def infer_problem_type(self, train_data: pd.DataFrame):
        return  # problem type should be provided in the learner initialization.

    def infer_output_shape(self):
        return  # learner doesn't need output shape since svm handles it.

    def prepare_fit_args(
        self,
        time_limit: int,
        seed: int,
        standalone: Optional[bool] = True,
        clean_ckpts: Optional[bool] = True,
    ):
        super().prepare_fit_args(
            time_limit=time_limit,
            seed=seed,
            standalone=standalone,
            clean_ckpts=clean_ckpts,
        )
        self._fit_args.pop("ckpt_path", None)
        self._fit_args.pop("resume", None)
        self._fit_args.pop("clean_ckpts", None)
        if self._fit_called:
            self._fit_args.update(dict(svm=self._svm))

    def fit_sanity_check(self):
        feature_column_types = {k: v for k, v in self._column_types.items() if k != self._label_column}
        unique_dtypes = set(feature_column_types.values())
        assert (
            len(unique_dtypes) == 1
        ), f"Few shot SVM learner allows single modality data for now, but detected modalities {unique_dtypes}."

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
        standalone: bool,
    ):
        self.clean_trainer_processes(trainer=trainer, is_train=True)
        self.save(
            path=save_path,
            standalone=standalone,
            config=config,
            model=model,
            svm=svm,
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
            fit_called=True,  # fit is called on one run.
            save_model=True,  # need to save the model now because there will be no top k averaging.
        )

    def get_datamodule_per_run(
        self,
        df_preprocessor,
        data_processors,
        per_gpu_batch_size,
        num_workers,
        predict_data=None,
        is_train=True,
    ):
        datamodule_kwargs = dict(
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
            per_gpu_batch_size=per_gpu_batch_size,
            num_workers=num_workers,
        )
        if is_train:
            datamodule_kwargs.update(dict(predict_data=self._train_data))
        else:
            datamodule_kwargs.update(dict(predict_data=predict_data))

        datamodule = BaseDataModule(**datamodule_kwargs)
        return datamodule

    @staticmethod
    def update_config_by_data_per_run(config, df_preprocessor):
        if df_preprocessor.text_feature_names and CLIP in config.model.names:
            config.model.names.remove(CLIP)  # for text only data, remove clip model and use hf_text
        if df_preprocessor.image_feature_names and TIMM_IMAGE in config.model.names and CLIP in config.model.names:
            config.model.names.remove(
                CLIP
            )  # if users add timm_image in hyperparameters, then remove clip and use timm_image
        config = select_model(config=config, df_preprocessor=df_preprocessor, strict=False)
        return config

    def init_trainer_per_run(
        self,
        num_gpus,
        precision,
        strategy,
        callbacks,
        max_time=None,
        config=None,
        enable_progress_bar=None,
        barebones=False,
        is_train=True,
    ):
        if not is_train:
            config = self._config
            enable_progress_bar = self._enable_progress_bar

        blacklist_msgs = []
        if self._verbosity <= 3:  # turn off logging in prediction
            blacklist_msgs.append("Automatic Mixed Precision")
            blacklist_msgs.append("GPU available")
            blacklist_msgs.append("TPU available")
            blacklist_msgs.append("IPU available")
            blacklist_msgs.append("HPU available")
            blacklist_msgs.append("select gpus")
            blacklist_msgs.append("Trainer(barebones=True)")
        log_filter = LogFilter(blacklist_msgs)

        with apply_log_filter(log_filter):
            trainer = pl.Trainer(
                accelerator="gpu" if num_gpus > 0 else "auto",
                devices=get_available_devices(num_gpus, config.env.auto_select_gpus),
                num_nodes=config.env.num_nodes,
                precision=precision,
                strategy=strategy,
                benchmark=False,
                enable_progress_bar=False if barebones else enable_progress_bar,
                deterministic=config.env.deterministic,
                max_epochs=-1,  # Add max_epochs to disable warning
                logger=False,
                callbacks=callbacks,
                barebones=barebones,
            )

        return trainer

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
        standalone: bool = True,
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
        svm = self.get_svm_per_run(svm=svm)
        if max_time == timedelta(seconds=0):
            return dict(
                config=config,
                df_preprocessor=df_preprocessor,
                data_processors=data_processors,
                model=model,
                svm=svm,
            )
        datamodule = self.get_datamodule_per_run(
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
            per_gpu_batch_size=config.env.per_gpu_batch_size,
            num_workers=config.env.num_workers,
        )
        num_gpus, strategy = self.get_num_gpus_and_strategy_per_run(config=config)
        precision = self.get_precision_per_run(num_gpus=num_gpus, precision=config.env.precision)
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
        features = self.aggregate_column_features(
            features=features,
            column_features_pooling_mode=config.data.column_features_pooling_mode,
        )
        # no need to call df_preprocessor.transform_label_for_metric since the sklearn pipeline encodes the label automatically
        labels = np.array(self._train_data[self._label_column])
        svm.fit(features, labels)
        self.on_fit_per_run_end(
            save_path=save_path,
            standalone=standalone,
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

    def aggregate_column_features(
        self,
        features: Union[Dict, np.ndarray],
        column_features_pooling_mode: Optional[str] = None,
        is_train: Optional[bool] = True,
    ):
        if not is_train:
            column_features_pooling_mode = self._config.data.column_features_pooling_mode

        if isinstance(features, np.ndarray):
            return features
        elif isinstance(features, dict):
            assert len(features) != 0, f"column features are empty."
            if len(features) == 1:
                return next(iter(features.values()))
            if column_features_pooling_mode == "concat":
                return np.concatenate(list(features.values()), axis=1)
            elif column_features_pooling_mode == "mean":
                return np.mean(list(features.values()), axis=0)
            else:
                raise ValueError(f"Unsupported column_features_pooling_mode: {column_features_pooling_mode}.")
        else:
            raise ValueError(f"Unsupported features type: {type(features)} in aggregating column features.")

    def predict(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        as_pandas: Optional[bool] = None,
        realtime: Optional[bool] = False,
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
            Whether to do realtime inference, which is efficient for small data (default False).
            If provided None, we would infer it on based on the data modalities
            and sample number.

        Returns
        -------
        Array of predictions, one corresponding to each row in given dataset.
        """
        self.on_predict_start()
        features = self.extract_embedding(data=data, realtime=realtime)
        pred = self._svm.predict(features)
        if (as_pandas is None and isinstance(data, pd.DataFrame)) or as_pandas is True:
            pred = self._as_pandas(data=data, to_be_converted=pred)
        return pred

    def predict_proba(
        self,
        data,
        as_pandas: Optional[bool] = False,
        as_multiclass: Optional[bool] = True,
        realtime: Optional[bool] = False,
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
            Whether to do realtime inference, which is efficient for small data (default False).
            If provided None, we would infer it on based on the data modalities
            and sample number.

        Returns
        -------
        Array of predicted class-probabilities, corresponding to each row in the given data.
        When as_multiclass is True, the output will always have shape (#samples, #classes).
        Otherwise, the output will have shape (#samples,)
        """
        self.on_predict_start()
        features = self.extract_embedding(data, realtime=realtime)
        logits = self._svm.decision_function(features)
        prob = logits_to_prob(logits)
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
            Whether to do realtime inference, which is efficient for small data (default False).
            If provided None, we would infer it on based on the data modalities
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
            realtime=realtime,
            requires_label=False,
        )
        features = extract_from_output(outputs=outputs, ret_type=COLUMN_FEATURES, as_ndarray=as_tensor is False)
        features = self.aggregate_column_features(features=features, is_train=False)
        return features

    def evaluate(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        metrics: Optional[Union[str, List[str]]] = None,
        return_pred: Optional[bool] = False,
        realtime: Optional[bool] = False,
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
            Whether to do realtime inference, which is efficient for small data (default False).
            If provided None, we would infer it on based on the data modalities
            and sample number.

        Returns
        -------
        A dictionary with the metric names and their corresponding scores.
        Optionally return a dataframe of prediction results.
        """
        self.on_predict_start()
        data = data_to_df(data=data)
        features = self.extract_embedding(data)
        pred = self._svm.predict(features)
        assert (
            self._label_column in data.columns
        ), f"Label {self._label_column} is not in the data. Cannot perform evaluation without ground truth labels."
        y_true = np.array(data[self._label_column])
        metric_data = {Y_PRED: pred, Y_TRUE: y_true}
        if metrics is None:
            if self._eval_metric_func:
                metrics = [self._eval_metric_func]
            else:
                metrics = [self._eval_metric_name]
        if isinstance(metrics, str) or isinstance(metrics, Scorer):
            metrics = [metrics]

        results = {}
        for per_metric in metrics:
            score = compute_score(
                metric_data=metric_data,
                metric=per_metric.lower() if isinstance(per_metric, str) else per_metric,
            )
            per_metric_name = per_metric if isinstance(per_metric, str) else per_metric.name
            results[per_metric_name] = score

        if return_pred:
            return results, self._as_pandas(data=data, to_be_converted=pred)
        return results

    @classmethod
    def load(
        cls,
        path: str,
        resume: Optional[bool] = False,
        verbosity: Optional[int] = 3,
    ):
        predictor = super().load(path=path, resume=resume, verbosity=verbosity)
        with open(os.path.join(path, "svm.pkl"), "rb") as fp:
            params = pickle.load(fp)  # nosec B301
        svm = make_pipeline(StandardScaler(), SVC(gamma="auto"))
        svm.set_params(**params)
        predictor._svm = svm

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
