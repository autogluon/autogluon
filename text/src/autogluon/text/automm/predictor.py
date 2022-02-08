import logging
import os
import numpy as np
import json
import warnings
import sys
import pandas as pd
import pickle
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torchmetrics
from omegaconf import OmegaConf, DictConfig
import pytorch_lightning as pl
from packaging import version
from typing import Optional, List, Tuple, Dict
from sklearn.model_selection import train_test_split
# from autogluon.core.utils import set_logger_verbosity
# from autogluon.core.utils.loaders import load_pd
from autogluon.core.utils.utils import (
    # setup_outputdir,
    default_holdout_frac,
)

from .constants import (
    BINARY, MULTICLASS, REGRESSION, Y_PRED,
    Y_PRED_PROB, Y_TRUE, LOGITS, FEATURES,
)

from .data.datamodule import BaseDataModule
from .data.infer_types import infer_column_problem_types
from .data.preprocess_dataframe import MultiModalFeaturePreprocessor

from .utils import (
    create_model,
    init_df_preprocessor,
    init_data_processors,
    select_model,
    compute_score,
    gather_top_k_ckpts,
    average_checkpoints,
    infer_eval_metric,
    get_config,
    setup_save_dir,
)
from .optimization.utils import (
    get_metric,
    get_loss_func,
)
from .optimization.lit_module import LitModule

from . import __version__

logger = logging.getLogger()  # return root logger


class AutoMMPredictor:

    def __init__(
            self,
            label: str,
            problem_type: Optional[str] = None,
            eval_metric: Optional[str] = None,
            path: Optional[str] = None,
            verbosity: Optional[int] = 3,
            warn_if_exist: Optional[bool] = True,
    ):
        self.verbosity = verbosity
        # if self.verbosity is not None:
        #     set_logger_verbosity(self.verbosity, logger=logger)

        self._label_column = label
        if eval_metric is not None and eval_metric.lower() in ["rmse", "r2"]:
            problem_type = REGRESSION

        self._problem_type = problem_type.lower() if problem_type is not None else None
        if eval_metric is None and problem_type is not None:
            eval_metric = infer_eval_metric(problem_type)

        self._eval_metric_name = eval_metric
        self._output_shape = None
        if path is not None:
            path = setup_save_dir(
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

    # def set_verbosity(self, verbosity: int):
    #     self.verbosity = verbosity
    #     set_logger_verbosity(self.verbosity, logger=logger)

    def fit(
            self,
            train_data: pd.DataFrame,
            config: Optional[dict] = None,
            tuning_data: Optional[pd.DataFrame] = None,
            save_path: Optional[str] = None,
            overrides=None,
            column_types=None,
            holdout_frac: Optional[float] = None,
            seed: Optional[int] = 123,
            init_only: Optional[bool] = False,
    ):
        pl.seed_everything(seed, workers=True)

        if self._config is None:
            config = get_config(
                config=config,
                overrides=overrides,
            )
        else:  # continuing training
            config = self._config

        if self._resume or save_path is None:
            save_path = self._save_path
        else:
            save_path = os.path.expanduser(save_path)

        if not self._resume:
            save_path = setup_save_dir(
                path=save_path,
                warn_if_exist=True,
            )
        print(f"save path: {save_path}")

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

        # set attributes for saving and prediction
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

        print(f"val_metric_name: {task.val_metric_name}")
        print(f"minmax_mode: {minmax_mode}")

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
        if num_gpus < 0:
            num_gpus = torch.cuda.device_count()

        grad_steps = config.env.batch_size // (
                config.env.per_gpu_batch_size * num_gpus * config.env.num_nodes
        )

        assert version.parse(pl.__version__) >= version.parse("1.5.9")
        print(f"deterministic: {config.env.deterministic}")
        trainer = pl.Trainer(
            gpus=config.env.num_gpus,
            auto_select_gpus=config.env.auto_select_gpus,
            num_nodes=config.env.num_nodes,
            precision=config.env.precision,
            strategy=config.env.strategy,
            benchmark=False,
            deterministic=config.env.deterministic,
            max_epochs=config.optimization.max_epochs,
            max_steps=config.optimization.max_steps,
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
            data: pd.DataFrame,
            ret_type: str,
    ) -> torch.Tensor:

        predict_dm = BaseDataModule(
            df_preprocessor=self._df_preprocessor,
            data_processors=self._data_processors,
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
        strategy = 'dp' if num_gpus > 1 else None
        evaluator = pl.Trainer(
            gpus=self._config.env.num_gpus,
            auto_select_gpus=self._config.env.auto_select_gpus,
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
        prob = prob.detach().cpu().numpy()
        return prob

    def evaluate(
            self,
            data: pd.DataFrame,
            metrics: Optional[List[str]] = None,
            return_pred: Optional[bool] = False,
    ):
        logits = self._predict(
            data=data,
            ret_type=LOGITS,
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
            if self._problem_type == MULTICLASS and per_metric.lower() == "roc_auc":
                raise ValueError(
                    "Problem type is multiclass, but roc_auc is only supported for binary classification."
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
            data: pd.DataFrame,
            as_pandas: Optional[bool] = True,
    ):

        logits = self._predict(
            data=data,
            ret_type=LOGITS,
        )
        pred = self._df_preprocessor.transform_prediction(y_pred=logits)
        if as_pandas:
            pred = self.as_pandas(data=data, to_be_converted=pred)
        return pred

    def predict_proba(
            self,
            data: pd.DataFrame,
            as_pandas: Optional[bool] = True,
            as_multiclass: Optional[bool] = True,
    ):

        assert self._problem_type in [BINARY, MULTICLASS], \
            f"Problem {self._problem_type} has no probability output"

        logits = self._predict(
            data=data,
            ret_type=LOGITS,
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
            data: pd.DataFrame,
            as_pandas: Optional[bool] = True,
    ):
        features = self._predict(
            data=data,
            ret_type=FEATURES,
        )
        features = features.detach().cpu().numpy()
        if as_pandas:
            features = pd.DataFrame(features, index=data.index)

        return features

    def as_pandas(
            self,
            data: pd.DataFrame,
            to_be_converted: np.ndarray,
    ):
        if to_be_converted.ndim == 1:
            return pd.Series(to_be_converted, index=data.index, name=self._label_column)
        else:
            return pd.DataFrame(to_be_converted, index=data.index, columns=self.class_labels)

    @staticmethod
    def _load_state_dict(model, state_dict):
        state_dict = {k.partition('model.')[2]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        return model

    def save(self, path: str):

        os.makedirs(path, exist_ok=True)
        OmegaConf.save(
            config=self._config,
            f=os.path.join(path, 'config.yaml')
        )

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
                    "version": __version__,
                },
                fp,
                ensure_ascii=True,
            )

    @classmethod
    def load(
            cls,
            path: str,
            resume: Optional[bool] = False,
    ):
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
            print(f"Resume training from checkpoint: '{resume_ckpt_path}'")
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
            print(f"Load pretrained checkpoint: {os.path.join(path, 'model.ckpt')}")
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
        """The original name of the class labels.
        For example, the tabular data may contain classes equal to
        "entailment", "contradiction", "neutral". Internally, these will be converted to
        0, 1, 2, ...
        This function returns the original names of these raw labels.
        Returns
        -------
        ret
            List that contain the class names. It will be None if it's not a classification problem.
        """
        if self._problem_type == MULTICLASS or self._problem_type == BINARY:
            return self._df_preprocessor.label_generator.classes_
        else:
            warnings.warn('Accessing class names for a non-classification problem. Return None.')
            return None

    @property
    def positive_class(self):
        """Name of the class label that will be mapped to 1. This is only meaningful for binary classification problems.

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

