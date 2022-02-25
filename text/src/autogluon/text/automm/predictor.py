import logging
import os
import numpy as np
import json
import warnings
import sys
import shutil
from datetime import timedelta
import pandas as pd
import pickle
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torchmetrics
from omegaconf import OmegaConf, DictConfig
import pytorch_lightning as pl
from typing import Optional, List, Tuple, Dict, Union
from sklearn.model_selection import train_test_split
from autogluon.core.utils.utils import default_holdout_frac
from autogluon.core.utils.loaders import load_pd
from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.common.utils.utils import setup_outputdir

from .constants import (
    LABEL, BINARY, MULTICLASS, REGRESSION, Y_PRED,
    Y_PRED_PROB, Y_TRUE, LOGITS, FEATURES, AUTOMM
)

from .data.datamodule import BaseDataModule
from .data.infer_types import infer_column_problem_types
from .data.preprocess_dataframe import MultiModalFeaturePreprocessor

from .utils import (
    create_model,
    create_and_save_model,
    init_df_preprocessor,
    init_data_processors,
    select_model,
    compute_score,
    gather_top_k_ckpts,
    average_checkpoints,
    infer_eval_metric,
    get_config,
)
from .optimization.utils import (
    get_metric,
    get_loss_func,
)
from .optimization.lit_module import LitModule

from .. import version

logger = logging.getLogger(AUTOMM)


class AutoMMPredictor:
    """
    AutoMMPredictor can predict the values of one dataframe column conditioned on the rest columns.
    The prediction can be either a classification or regression problem. The feature columns can contain
    image paths, text, numerical, and categorical features.
    """

    def __init__(
            self,
            label: str,
            problem_type: Optional[str] = None,
            eval_metric: Optional[str] = None,
            path: Optional[str] = None,
            verbosity: Optional[int] = 3,
            warn_if_exist: Optional[bool] = True,
    ):
        """
        Parameters
        ----------
        label
            Name of the column that contains the target variable to predict.
        problem_type
            Type of prediction problem, i.e. is this a binary/multiclass classification or regression problem
            (options: 'binary', 'multiclass', 'regression').
            If `problem_type = None`, the prediction problem type is inferred
            based on the label-values in provided dataset.
        eval_metric
            Evaluation metric name. If `eval_metric = None`, it is automatically chosen based on `problem_type`.
            Defaults to 'accuracy' for binary and multiclass classification, 'root_mean_squared_error' for regression.
        path
            Path to directory where models and intermediate outputs should be saved.
            If unspecified, a time-stamped folder called "AutogluonAutoMM/ag-[TIMESTAMP]"
            will be created in the working directory to store all models.
            Note: To call `fit()` twice and save all results of each fit,
            you must specify different `path` locations or don't specify `path` at all.
            Otherwise files from first `fit()` will be overwritten by second `fit()`.
        verbosity
            Verbosity levels range from 0 to 4 and control how much information is printed.
            Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
            If using logging, you can alternatively control amount of information printed via `logger.setLevel(L)`,
            where `L` ranges from 0 to 50
            (Note: higher values of `L` correspond to fewer print statements, opposite of verbosity levels)
        warn_if_exist
            Whether to raise warning if the specified path already exists.
        """
        self.verbosity = verbosity
        if self.verbosity is not None:
            set_logger_verbosity(self.verbosity, logger=logger)

        self._label_column = label

        if eval_metric is not None and not isinstance(eval_metric, str):
            eval_metric = eval_metric.name

        if eval_metric is not None and eval_metric.lower() in ["rmse", "r2"]:
            problem_type = REGRESSION

        self._problem_type = problem_type.lower() if problem_type is not None else None
        if eval_metric is None and problem_type is not None:
            eval_metric = infer_eval_metric(problem_type)

        self._eval_metric_name = eval_metric
        self._output_shape = None
        if path is not None:
            path = setup_outputdir(
                path=path,
                warn_if_exist=warn_if_exist,
            )
        self._save_path = path
        self._ckpt_path = None
        self._pretrained_path = None
        self._config = None
        self._df_preprocessor = None
        self._column_types = None
        self._data_processors = None
        self._model = None
        self._resume = False

    @property
    def path(self):
        return self._save_path

    @property
    def label(self):
        return self._label_column

    @property
    def problem_type(self):
        return self._problem_type

    def set_verbosity(self, verbosity: int):
        self.verbosity = verbosity
        set_logger_verbosity(self.verbosity, logger=logger)

    def fit(
            self,
            train_data: pd.DataFrame,
            config: Optional[dict] = None,
            tuning_data: Optional[pd.DataFrame] = None,
            time_limit: Optional[int] = None,
            save_path: Optional[str] = None,
            hyperparameters: Optional[Union[str, Dict, List[str]]] = None,
            column_types: Optional[dict] = None,
            holdout_frac: Optional[float] = None,
            seed: Optional[int] = 123,
            init_only: Optional[bool] = False,
    ):
        """
        Fit AutoMMPredictor predict label column of a dataframe based on the other columns,
        which may contain image path, text, numeric, or categorical features.

        Parameters
        ----------
        train_data
            A dataframe containing training data.
        config
            A dictionary with four keys "model", "data", "optimization", and "environment".
            Each key's value can be a string, yaml file path, or OmegaConf's DictConfig.
            Strings should be the file names (DO NOT include the postfix ".yaml") in
            automm/configs/model, automm/configs/data, automm/configs/optimization, and automm/configs/environment.
            For example, you can configure a late-fusion model for the image, text, and tabular data as follows:
            config = {
                        "model": "fusion_mlp_image_text_tabular",
                        "data": "default",
                        "optimization": "adamw",
                        "environment": "default",
                    }
            or
            config = {
                        "model": "/path/to/model/config.yaml",
                        "data": "/path/to/data/config.yaml",
                        "optimization": "/path/to/optimization/config.yaml",
                        "environment": "/path/to/environment/config.yaml",
                    }
            or
            config = {
                        "model": OmegaConf.load("/path/to/model/config.yaml"),
                        "data": OmegaConf.load("/path/to/data/config.yaml"),
                        "optimization": OmegaConf.load("/path/to/optimization/config.yaml"),
                        "environment": OmegaConf.load("/path/to/environment/config.yaml"),
                    }
        tuning_data
            A dataframe containing validation data, which should have the same columns as the train_data.
            If `tuning_data = None`, `fit()` will automatically
            hold out some random validation examples from `train_data`.
        time_limit
            How long `fit()` should run for (wall clock time in seconds).
            If not specified, `fit()` will run until the model has completed training.
        save_path
            Path to directory where models and intermediate outputs should be saved.
        hyperparameters
            This is to override some default configurations.
            For example, changing the text and image backbones can be done by formatting:

            a string
            hyperparameters = "model.hf_text.checkpoint_name=google/electra-small-discriminator model.timm_image.checkpoint_name=swin_small_patch4_window7_224"

            or a list of strings
            hyperparameters = ["model.hf_text.checkpoint_name=google/electra-small-discriminator", "model.timm_image.checkpoint_name=swin_small_patch4_window7_224"]

            or a dictionary
            hyperparameters = {
                            "model.hf_text.checkpoint_name": "google/electra-small-discriminator",
                            "model.timm_image.checkpoint_name": "swin_small_patch4_window7_224",
                        }
        column_types
            A dictionary that maps column names to their data types.
            For example: `column_types = {"item_name": "text", "image": "image_path",
            "product_description": "text", "height": "numerical"}`
            may be used for a table with columns: "item_name", "brand", "product_description", and "height".
            If None, column_types will be automatically inferred from the data.
            The current supported types are:
                - "image_path": each row in this column is one image path.
                - "text": each row in this column contains text (sentence, paragraph, etc.).
                - "numerical": each row in this column contains a number.
                - "categorical": each row in this column belongs to one of K categories.
        holdout_frac
            Fraction of train_data to holdout as tuning_data for optimizing hyper-parameters or
            early stopping (ignored unless `tuning_data = None`).
            Default value (if None) is selected based on the number of rows in the training data
            and whether hyper-parameter-tuning is utilized.
        seed
            The random seed to use for this training run.
        init_only
            Whether to only initialize the model without training. This can be used when we want to
            compare the model performance before and after training.
        Returns
        -------
        An "AutoMMPredictor" object (itself).
        """
        pl.seed_everything(seed, workers=True)

        if self._config is None:
            config = get_config(
                config=config,
                overrides=hyperparameters,
            )
        else:  # continuing training
            config = self._config

        if self._resume or save_path is None:
            save_path = self._save_path
        else:
            save_path = os.path.expanduser(save_path)

        if not self._resume:
            save_path = setup_outputdir(
                path=save_path,
                warn_if_exist=True,
            )
        logger.debug(f"save path: {save_path}")

        if tuning_data is None:
            if self._problem_type in [BINARY, MULTICLASS]:
                stratify = train_data[self._label_column]
            else:
                stratify = None
            if holdout_frac is None:
                val_frac = default_holdout_frac(len(train_data), hyperparameter_tune=False)
            else:
                val_frac = holdout_frac
            train_data, tuning_data = train_test_split(
                train_data,
                test_size=val_frac,
                stratify=stratify,
                random_state=np.random.RandomState(seed)
            )

        inferred_column_types, problem_type, output_shape = \
            infer_column_problem_types(
                train_df=train_data,
                valid_df=tuning_data,
                label_columns=self._label_column,
                problem_type=self._problem_type,
            )

        if column_types is None:
            column_types = inferred_column_types

        logger.debug(f"column_types: {column_types}")
        logger.debug(f"image columns: {[k for k, v in column_types.items() if v == 'image_path']}")

        if self._column_types is not None and self._column_types != column_types:
            warnings.warn(
                f"Inferred column types {column_types} are inconsistent with "
                f"the previous {self._column_types}. "
                f"New columns will not be used in the current training."
            )
            # use previous column types to avoid inconsistency with previous numerical mlp and categorical mlp
            column_types = self._column_types

        if self._problem_type is not None:
            assert self._problem_type == problem_type, \
                f"Inferred problem type {problem_type} is different from " \
                f"the previous {self._problem_type}"

        if self._output_shape is not None:
            assert self._output_shape == output_shape, \
                f"Inferred output shape {output_shape} is different from " \
                f"the previous {self._output_shape}"

        if self._eval_metric_name is None:
            self._eval_metric_name = infer_eval_metric(problem_type)

        if self._df_preprocessor is None:
            df_preprocessor = init_df_preprocessor(
                config=config.data,
                column_types=column_types,
                label_column=self._label_column,
                train_df_x=train_data.drop(columns=self._label_column),
                train_df_y=train_data[self._label_column]
            )
        else:  # continuing training
            df_preprocessor = self._df_preprocessor

        config = select_model(
            config=config,
            df_preprocessor=df_preprocessor
        )

        if self._data_processors is None:
            data_processors = init_data_processors(
                config=config,
                num_categorical_columns=len(df_preprocessor.categorical_num_categories)
            )
        else:  # continuing training
            data_processors = self._data_processors

        if self._model is None:
            model = create_model(
                config=config,
                num_classes=output_shape,
                num_numerical_columns=len(df_preprocessor.numerical_feature_names),
                num_categories=df_preprocessor.categorical_num_categories
            )
        else:  # continuing training
            model = self._model

        val_metric, minmax_mode = get_metric(
            metric_name=self._eval_metric_name,
            num_classes=output_shape
        )
        loss_func = get_loss_func(problem_type)

        if time_limit is not None:
            time_limit = timedelta(seconds=time_limit)

        # set attributes for saving and prediction
        self._problem_type = problem_type  # In case problem wasn't provided in __init__().
        self._save_path = save_path
        self._config = config
        self._output_shape = output_shape
        self._column_types = column_types
        self._df_preprocessor = df_preprocessor
        self._data_processors = data_processors
        self._model = model

        # save artifacts for the current running, except for model checkpoint, which will be saved in _fit()
        self.save(save_path)

        if init_only:
            return self

        self._fit(
            train_df=train_data,
            val_df=tuning_data,
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
            model=model,
            config=config,
            loss_func=loss_func,
            val_metric=val_metric,
            minmax_mode=minmax_mode,
            max_time=time_limit,
            save_path=save_path,
        )
        return self

    def _fit(
            self,
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
            df_preprocessor: MultiModalFeaturePreprocessor,
            data_processors: dict,
            model: nn.Module,
            config: DictConfig,
            loss_func: _Loss,
            val_metric: torchmetrics.Metric,
            minmax_mode: str,
            max_time: timedelta,
            save_path: str,
    ):

        train_dm = BaseDataModule(
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
            per_gpu_batch_size=config.env.per_gpu_batch_size,
            num_workers=config.env.num_workers,
            train_data=train_df,
            val_data=val_df
        )

        task = LitModule(
            model=model,
            optim_type=config.optimization.optim_type,
            lr_choice=config.optimization.lr_choice,
            lr_schedule=config.optimization.lr_schedule,
            lr=config.optimization.learning_rate,
            lr_decay=config.optimization.lr_decay,
            end_lr=config.optimization.end_lr,
            lr_mult=config.optimization.lr_mult,
            weight_decay=config.optimization.weight_decay,
            warmup_steps=config.optimization.warmup_steps,
            loss_func=loss_func,
            val_metric=val_metric,
        )

        logger.debug(f"val_metric_name: {task.val_metric_name}")
        logger.debug(f"minmax_mode: {minmax_mode}")

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=save_path,
            save_top_k=config.optimization.top_k,
            verbose=True,
            monitor=task.val_metric_name,
            mode=minmax_mode,
            save_last=True,
        )
        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor=task.val_metric_name,
            patience=config.optimization.patience,
            mode=minmax_mode
        )
        lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
        model_summary = pl.callbacks.ModelSummary(max_depth=1)
        callbacks = [checkpoint_callback, early_stopping_callback, lr_callback, model_summary]

        tb_logger = pl.loggers.TensorBoardLogger(
            save_dir=save_path,
            name="",
            version="",
        )

        num_gpus = (
            config.env.num_gpus
            if isinstance(config.env.num_gpus, int)
            else len(config.env.num_gpus)
        )
        if num_gpus < 0:  # In case config.env.num_gpus is -1, meaning using all gpus.
            num_gpus = torch.cuda.device_count()

        if num_gpus == 0:  # CPU only training
            warnings.warn(
                "Only CPU is detected in the instance. "
                "AutoMMPredictor will be trained with CPU only. "
                "This may results in slow training speed. "
                "Consider to switch to an instance with GPU support.",
                UserWarning,
            )
            grad_steps = config.env.batch_size // (
                    config.env.per_gpu_batch_size * config.env.num_nodes
            )
        else:
            grad_steps = config.env.batch_size // (
                    config.env.per_gpu_batch_size * num_gpus * config.env.num_nodes
            )

        if num_gpus <= 1:
            strategy = None
        else:
            strategy = config.env.strategy

        trainer = pl.Trainer(
            gpus=num_gpus,
            auto_select_gpus=config.env.auto_select_gpus if num_gpus != 0 else False,
            num_nodes=config.env.num_nodes,
            precision=config.env.precision,
            strategy=strategy,
            benchmark=False,
            deterministic=config.env.deterministic,
            max_epochs=config.optimization.max_epochs,
            max_steps=config.optimization.max_steps,
            max_time=max_time,
            callbacks=callbacks,
            logger=tb_logger,
            gradient_clip_val=1,
            gradient_clip_algorithm="norm",
            accumulate_grad_batches=grad_steps,
            log_every_n_steps=10,
            fast_dev_run=config.env.fast_dev_run,
            val_check_interval=config.optimization.val_check_interval,
        )
        trainer.fit(
            task,
            datamodule=train_dm,
            ckpt_path=self._ckpt_path,  # this is to resume training that was broken accidentally
        )

        if trainer.global_rank == 0:
            top_k_avg_ckpt_path = os.path.join(save_path, "model.ckpt")
            all_state_dicts, ckpt_template = gather_top_k_ckpts(
                ckpt_dir=save_path,
                ckpt_paths=checkpoint_callback.best_k_models.keys(),
            )
            avg_state_dict = average_checkpoints(
                all_state_dicts=all_state_dicts,
                out_path=top_k_avg_ckpt_path,
                ckpt_template=ckpt_template,
            )
            model = self._load_state_dict(
                model=model,
                state_dict=avg_state_dict,
            )

            self._model = model
        else:
            sys.exit(
                f"Training finished, exit the process with global_rank={trainer.global_rank}..."
            )

    def _predict(
            self,
            data: Union[pd.DataFrame, dict, list],
            ret_type: str,
            requires_label: bool,
    ) -> torch.Tensor:

        data = self._data_to_df(data)

        data_processors = self._data_processors.copy()
        # For prediction data with no labels provided.
        if not requires_label:
            data_processors.pop(LABEL, None)
        logger.debug(f"data_processors for prediction: {data_processors.keys()}")

        predict_dm = BaseDataModule(
            df_preprocessor=self._df_preprocessor,
            data_processors=data_processors,
            per_gpu_batch_size=self._config.env.per_gpu_batch_size,
            num_workers=self._config.env.num_workers_evaluation,
            predict_data=data,
        )
        task = LitModule(
            model=self._model,
        )
        num_gpus = (
            self._config.env.num_gpus
            if isinstance(self._config.env.num_gpus, int)
            else len(self._config.env.num_gpus)
        )
        if num_gpus < 0:
            num_gpus = torch.cuda.device_count()

        if num_gpus == 0:  # CPU only prediction
            warnings.warn(
                "Only CPU is detected in the instance. "
                "AutoMMPredictor will predict with CPU only. "
                "This may results in slow prediction speed. "
                "Consider to switch to an instance with GPU support.",
                UserWarning,
            )

        strategy = 'dp' if num_gpus > 1 else None
        evaluator = pl.Trainer(
            gpus=num_gpus,
            auto_select_gpus=self._config.env.auto_select_gpus if num_gpus != 0 else False,
            num_nodes=self._config.env.num_nodes,
            precision=self._config.env.precision,
            strategy=strategy,
            benchmark=False,
            deterministic=self._config.env.deterministic,
            logger=False,
        )

        outputs = evaluator.predict(
            task,
            datamodule=predict_dm,
        )
        if ret_type == LOGITS:
            logits = [ele[LOGITS] for ele in outputs]
            ret = torch.cat(logits)
        elif ret_type == FEATURES:
            features = [ele[FEATURES] for ele in outputs]
            ret = torch.cat(features)
        else:
            raise ValueError(f"Unknown return type: {ret_type}")

        return ret

    @staticmethod
    def _logits_to_prob(logits: torch.Tensor):
        assert logits.ndim == 2
        prob = F.softmax(logits.float(), dim=1)
        prob = prob.detach().cpu().float().numpy()
        return prob

    def evaluate(
            self,
            data: Union[pd.DataFrame, dict, list],
            metrics: Optional[List[str]] = None,
            return_pred: Optional[bool] = False,
    ):
        """
        Evaluate model on a test dataset.

        Parameters
        ----------
        data
            A dataframe, containing the same columns as the training data
        metrics
            A list of metric names to report.
            If None, we only return the score for the stored `_eval_metric_name`.
        return_pred
            Whether to return the prediction result of each row.

        Returns
        -------
        A dictionary with the metric names and their corresponding scores.
        Optionally return a dataframe of prediction results.
        """
        logits = self._predict(
            data=data,
            ret_type=LOGITS,
            requires_label=True,
        )
        metric_data = {}
        if self._problem_type in [BINARY, MULTICLASS]:
            y_pred_prob = self._logits_to_prob(logits)
            metric_data[Y_PRED_PROB] = y_pred_prob

        y_pred = self._df_preprocessor.transform_prediction(y_pred=logits)
        y_true = self._df_preprocessor.transform_label_for_metric(df=data)

        metric_data.update({
            Y_PRED: y_pred,
            Y_TRUE: y_true,
        })

        if metrics is None:
            metrics = [self._eval_metric_name]

        results = {}
        for per_metric in metrics:
            if self._problem_type != BINARY and per_metric.lower() in ["roc_auc", "average_precision"]:
                raise ValueError(
                    f"Metric {per_metric} is only supported for binary classification."
                )
            score = compute_score(
                metric_data=metric_data,
                metric_name=per_metric,
            )
            results[per_metric] = score

        if return_pred:
            return results, self.as_pandas(data=data, to_be_converted=y_pred)
        else:
            return results

    def predict(
            self,
            data: Union[pd.DataFrame, dict, list],
            as_pandas: Optional[bool] = True,
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

        Returns
        -------
        Array of predictions, one corresponding to each row in given dataset.
        """
        logits = self._predict(
            data=data,
            ret_type=LOGITS,
            requires_label=False,
        )
        pred = self._df_preprocessor.transform_prediction(y_pred=logits)
        if as_pandas:
            pred = self.as_pandas(data=data, to_be_converted=pred)
        return pred

    def predict_proba(
            self,
            data: Union[pd.DataFrame, dict, list],
            as_pandas: Optional[bool] = True,
            as_multiclass: Optional[bool] = True,
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

        Returns
        -------
        Array of predicted class-probabilities, corresponding to each row in the given data.
        When as_multiclass is True, the output will always have shape (#samples, #classes).
        Otherwise, the output will have shape (#samples,)
        """
        assert self._problem_type in [BINARY, MULTICLASS], \
            f"Problem {self._problem_type} has no probability output."

        logits = self._predict(
            data=data,
            ret_type=LOGITS,
            requires_label=False,
        )
        prob = self._logits_to_prob(logits)

        if not as_multiclass:
            if self._problem_type == BINARY:
                prob = prob[:, 1]
        if as_pandas:
            prob = self.as_pandas(data=data, to_be_converted=prob)
        return prob

    def extract_embedding(
            self,
            data: Union[pd.DataFrame, dict, list],
            as_pandas: Optional[bool] = False,
    ):
        """
        Extract features for each sample, i.e., one row in the provided dataframe `data`.

        Parameters
        ----------
        data
            The data to extract embeddings for. Should contain same column names as training dataset and
            follow same format (except for the `label` column).
        as_pandas
            Whether to return the output as a pandas DataFrame (True) or numpy array (False).

        Returns
        -------
        Array of embeddings, corresponding to each row in the given data.
        It will have shape (#samples, D) where the embedding dimension D is determined
        by the neural network's architecture.
        """
        features = self._predict(
            data=data,
            ret_type=FEATURES,
            requires_label=False,
        )
        features = features.detach().cpu().numpy()
        if as_pandas:
            features = pd.DataFrame(features, index=data.index)

        return features

    def _data_to_df(self, data: Union[pd.DataFrame, dict, list]):
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, (list, dict)):
            data = pd.DataFrame(data)
        elif isinstance(data, str):
            data = load_pd.load(data)
        else:
            raise NotImplementedError(
                f'The format of data is not understood. '
                f'We have type(data)="{type(data)}", but a pd.DataFrame was required.'
            )
        return data

    def as_pandas(
            self,
            data: Union[pd.DataFrame, dict, list],
            to_be_converted: np.ndarray,
    ):
        if isinstance(data, pd.DataFrame):
            index = data.index
        else:
            index = None
        if to_be_converted.ndim == 1:
            return pd.Series(to_be_converted, index=index, name=self._label_column)
        else:
            return pd.DataFrame(to_be_converted, index=index, columns=self.class_labels)

    @staticmethod
    def _load_state_dict(model, state_dict):
        state_dict = {k.partition('model.')[2]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        return model

    def save(self, path: str, standalone: Optional[bool] = False,):
        """
        Save this predictor to file in directory specified by `path`.

        Parameters
        ----------
        path
            The directory to save this predictor.
        """
        os.makedirs(path, exist_ok=True)
        OmegaConf.save(
            config=self._config,
            f=os.path.join(path, 'config.yaml')
        )
        
        if standalone:
            model = create_and_save_model(
                config=self._config,
                num_classes=self._output_shape,
                save_path=save_path,
                num_numerical_columns=len(self._df_preprocessor.numerical_feature_names),
                num_categories=self._df_preprocessor.categorical_num_categories
            )

            for model_name in self._config.model.names:
                if "hf_text" in model_name or model_name == "clip":
                    model_config = getattr(self._config.model, model_name)
                    model_config.checkpoint_name = os.path.join(save_path,model_name)
                    
        with open(os.path.join(path, "df_preprocessor.pkl"), "wb") as fp:
            pickle.dump(self._df_preprocessor, fp)

        with open(os.path.join(path, "data_processors.pkl"), "wb") as fp:
            pickle.dump(self._data_processors, fp)

        with open(os.path.join(path, f"assets.json"), "w") as fp:
            json.dump(
                {
                    "column_types": self._column_types,
                    "label_column": self._label_column,
                    "problem_type": self._problem_type,
                    "eval_metric_name": self._eval_metric_name,
                    "output_shape": self._output_shape,
                    "save_path": self._save_path,
                    "pretrained_path": self._pretrained_path,
                    "version": version.__version__,
                },
                fp,
                ensure_ascii=True,
            )
        # In case that users save to a path, which is not the original save_path.
        if path != self._save_path:
            shutil.copy(os.path.join(self._save_path, "model.ckpt"), path)

    @classmethod
    def load(
            cls,
            path: str,
            resume: Optional[bool] = False,
            verbosity: Optional[int] = 3,
    ):
        """
        Load a predictor object from a directory specified by `path`. The to-be-loaded predictor
        can be completely or partially trained by .fit(). If a previous training has completed,
        it will load the checkpoint `model.ckpt`. Otherwise if a previous training accidentally
        collapses in the middle, it can load the `last.ckpt` checkpoint by setting `resume=True`.

        Parameters
        ----------
        path
            The directory to load the predictor object.
        resume
            Whether to resume training from `last.ckpt`. This is useful when a training was accidentally
            broken during the middle and we want to resume the training from the last saved checkpoint.
        verbosity
            Verbosity levels range from 0 to 4 and control how much information is printed.
            Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).

        Returns
        -------
        The loaded predictor object.
        """
        path = os.path.expanduser(path)
        assert os.path.isdir(path), f"'{path}' must be an existing directory."
        config = OmegaConf.load(os.path.join(path, "config.yaml"))
        with open(os.path.join(path, "df_preprocessor.pkl"), "rb") as fp:
            df_preprocessor = pickle.load(fp)
        with open(os.path.join(path, "data_processors.pkl"), "rb") as fp:
            data_processors = pickle.load(fp)
        with open(os.path.join(path, "assets.json"), "r") as fp:
            assets = json.load(fp)

        predictor = cls(
            label=assets["label_column"],
            problem_type=assets["problem_type"],
            eval_metric=assets["eval_metric_name"],
            verbosity=verbosity,
        )
        predictor._resume = resume
        predictor._save_path = path  # in case the original exp dir is copied to somewhere else
        predictor._pretrain_path = path
        predictor._config = config
        predictor._output_shape = assets["output_shape"]
        predictor._column_types = assets["column_types"]
        predictor._df_preprocessor = df_preprocessor
        predictor._data_processors = data_processors

        model = create_model(
            config=config,
            num_classes=assets["output_shape"],
            num_numerical_columns=len(df_preprocessor.numerical_feature_names),
            num_categories=df_preprocessor.categorical_num_categories
        )

        resume_ckpt_path = os.path.join(path, "last.ckpt")
        final_ckpt_path = os.path.join(path, "model.ckpt")
        if resume:  # resume training which crashed before
            if not os.path.isfile(resume_ckpt_path):
                if os.path.isfile(final_ckpt_path):
                    raise ValueError(
                        f"Resuming checkpoint '{resume_ckpt_path}' doesn't exist, but "
                        f"final checkpoint '{final_ckpt_path}' exists, which means training "
                        f"is already completed."
                    )
                else:
                    raise ValueError(
                        f"Resuming checkpoint '{resume_ckpt_path}' and "
                        f"final checkpoint '{final_ckpt_path}' both don't exist. "
                        f"Consider starting training from scratch."
                    )
            checkpoint = torch.load(resume_ckpt_path)
            logger.info(f"Resume training from checkpoint: '{resume_ckpt_path}'")
            ckpt_path = resume_ckpt_path
        else:  # load a model checkpoint for prediction, evaluation, or continuing training on new data
            if not os.path.isfile(final_ckpt_path):
                if os.path.isfile(resume_ckpt_path):
                    raise ValueError(
                        f"Final checkpoint '{final_ckpt_path}' doesn't exist, but "
                        f"resuming checkpoint '{resume_ckpt_path}' exists, which means training "
                        f"is not done yet. Consider resume training from '{resume_ckpt_path}'."
                    )
                else:
                    raise ValueError(
                        f"Resuming checkpoint '{resume_ckpt_path}' and "
                        f"final checkpoint '{final_ckpt_path}' both don't exist. "
                        f"Consider starting training from scratch."
                    )
            checkpoint = torch.load(final_ckpt_path)
            logger.info(f"Load pretrained checkpoint: {os.path.join(path, 'model.ckpt')}")
            ckpt_path = None  # must set None since we do not resume training

        model = cls._load_state_dict(
            model=model,
            state_dict=checkpoint["state_dict"],
        )

        predictor._ckpt_path = ckpt_path
        predictor._model = model

        return predictor

    @property
    def class_labels(self):
        """
        The original name of the class labels.
        For example, the tabular data may contain classes equal to
        "entailment", "contradiction", "neutral". Internally, these will be converted to
        0, 1, 2, ...
        This function returns the original names of these raw labels.

        Returns
        -------
        List that contain the class names. It will be None if it's not a classification problem.
        """
        if self._problem_type == MULTICLASS or self._problem_type == BINARY:
            return self._df_preprocessor.label_generator.classes_
        else:
            warnings.warn('Accessing class names for a non-classification problem. Return None.')
            return None

    @property
    def positive_class(self):
        """
        Name of the class label that will be mapped to 1.
        This is only meaningful for binary classification problems.

        It is useful for computing metrics such as F1 which require a positive and negative class.
        You may refer to https://en.wikipedia.org/wiki/F-score for more details.
        In binary classification, :class:`TextPredictor.predict_proba(as_multiclass=False)`
        returns the estimated probability that each row belongs to the positive class.
        Will print a warning and return None if called when `predictor.problem_type != 'binary'`.

        Returns
        -------
        The positive class name in binary classification or None if the problem is not binary classification.
        """
        if self.problem_type != BINARY:
            logger.warning(
                f"Warning: Attempted to retrieve positive class label in a non-binary problem. "
                f"Positive class labels only exist in binary classification. "
                f"Returning None instead. self.problem_type is '{self.problem_type}'"
                f" but positive_class only exists for '{BINARY}'.")
            return None
        else:
            return self.class_labels[1]
