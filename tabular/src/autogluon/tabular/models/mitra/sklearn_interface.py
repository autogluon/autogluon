from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from ._internal.config.config_run import ConfigRun
from ._internal.config.enums import ModelName
from ._internal.core.trainer_finetune import TrainerFinetune
from ._internal.data.dataset_split import make_stratified_dataset_split
from ._internal.models.tab2d import Tab2D
from ._internal.utils.set_seed import set_seed

# Hyperparameter search space
DEFAULT_FINE_TUNE = True # [True, False]
DEFAULT_FINE_TUNE_STEPS = 50 # [50, 60, 70, 80, 90, 100]
DEFAULT_CLS_METRIC = 'log_loss' # ['log_loss', 'accuracy', 'auc']
DEFAULT_REG_METRIC = 'mse' # ['mse', 'mae', 'rmse', 'r2']
SHUFFLE_CLASSES = False # [True, False]
SHUFFLE_FEATURES = False # [True, False]
USE_RANDOM_TRANSFORMS = False # [True, False]
RANDOM_MIRROR_REGRESSION = True # [True, False]
RANDOM_MIRROR_X = True # [True, False]
LR = 0.0001 # [0.00001, 0.000025, 0.00005, 0.000075, 0.0001, 0.00025, 0.0005, 0.00075, 0.001]
PATIENCE = 40 # [30, 35, 40, 45, 50]
WARMUP_STEPS = 1000 # [500, 750, 1000, 1250, 1500]
DEFAULT_GENERAL_MODEL = 'autogluon/mitra-classifier'
DEFAULT_CLS_MODEL = 'autogluon/mitra-classifier'
DEFAULT_REG_MODEL = 'autogluon/mitra-regressor'

# Constants
SEED = 0
DEFAULT_MODEL_TYPE = "Tab2D"

def _get_default_device():
    """Get the best available device for the current system."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"  # Apple silicon
    else:
        return "cpu"

DEFAULT_DEVICE = _get_default_device()
DEFAULT_ENSEMBLE = 1
DEFAULT_DIM = 512
DEFAULT_LAYERS = 12
DEFAULT_HEADS = 4
DEFAULT_CLASSES = 10
DEFAULT_VALIDATION_SPLIT = 0.2
USE_HF = True  # Use Hugging Face pretrained models if available

class MitraBase(BaseEstimator):
    """Base class for Mitra models with common functionality."""

    def __init__(self,
            model_type=DEFAULT_MODEL_TYPE,
            n_estimators=DEFAULT_ENSEMBLE,
            device=DEFAULT_DEVICE,
            fine_tune=DEFAULT_FINE_TUNE,
            fine_tune_steps=DEFAULT_FINE_TUNE_STEPS,
            metric=DEFAULT_CLS_METRIC,
            state_dict=None,
            hf_general_model=DEFAULT_GENERAL_MODEL,
            hf_cls_model=DEFAULT_CLS_MODEL,
            hf_reg_model=DEFAULT_REG_MODEL,
            patience=PATIENCE,
            lr=LR,
            warmup_steps=WARMUP_STEPS,
            shuffle_classes=SHUFFLE_CLASSES,
            shuffle_features=SHUFFLE_FEATURES,
            use_random_transforms=USE_RANDOM_TRANSFORMS,
            random_mirror_regression=RANDOM_MIRROR_REGRESSION,
            random_mirror_x=RANDOM_MIRROR_X,
            seed=SEED,
            verbose=True,
        ):
        """
        Initialize the base Mitra model.
        
        Parameters
        ----------
        model_type : str, default="Tab2D"
            The type of model to use. Options: "Tab2D", "Tab2D_COL_ROW"
        n_estimators : int, default=1
            Number of models in the ensemble
        device : str, default="cuda"
            Device to run the model on
        fine_tune_steps: int, default=0
            Number of epochs to train for
        state_dict : str, optional
            Path to the pretrained weights
        """
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.device = device
        self.fine_tune = fine_tune
        self.fine_tune_steps = fine_tune_steps
        self.metric = metric
        self.state_dict = state_dict
        self.hf_general_model = hf_general_model
        self.hf_cls_model = hf_cls_model
        self.hf_reg_model = hf_reg_model
        self.patience = patience
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.shuffle_classes = shuffle_classes
        self.shuffle_features = shuffle_features
        self.use_random_transforms = use_random_transforms
        self.random_mirror_regression = random_mirror_regression
        self.random_mirror_x = random_mirror_x
        self.trainers = []
        self.train_time = 0
        self.seed = seed
        self.verbose = verbose

        # FIXME: set_seed was removed in v1.4 as quality and speed reduction was observed when setting seed.
        #  This should be investigated and fixed for v1.5
        # set_seed(self.seed)

    def _create_config(self, task, dim_output, time_limit=None):
        cfg = ConfigRun(
            device=self.device,
            model_name=ModelName.TAB2D,
            seed=self.seed,
            hyperparams={
                'dim_embedding': None,
                'early_stopping_data_split': 'VALID',
                'early_stopping_max_samples': 2048,
                'early_stopping_patience': self.patience,
                'grad_scaler_enabled': False,
                'grad_scaler_growth_interval': 1000,
                'grad_scaler_scale_init': 65536.0,
                'grad_scaler_scale_min': 65536.0,
                'label_smoothing': 0.0,
                'lr_scheduler': False,
                'lr_scheduler_patience': 25,
                'max_epochs': self.fine_tune_steps if self.fine_tune else 0,
                'max_samples_query': 1024,
                'max_samples_support': 8192,
                'optimizer': 'adamw',
                'lr': self.lr,
                'weight_decay': 0.1,
                'warmup_steps': self.warmup_steps,
                'path_to_weights': self.state_dict,
                'precision': 'bfloat16',
                'random_mirror_regression': self.random_mirror_regression,
                'random_mirror_x': self.random_mirror_x,
                'shuffle_classes': self.shuffle_classes,
                'shuffle_features': self.shuffle_features,
                'use_random_transforms': self.use_random_transforms,
                'use_feature_count_scaling': False,
                'use_pretrained_weights': False,
                'use_quantile_transformer': False,
                'budget': time_limit,
                'metric': self.metric,
            },
        )

        cfg.task = task
        cfg.hyperparams.update({
            'n_ensembles': self.n_estimators,
            'dim': DEFAULT_DIM,
            'dim_output': dim_output,
            'n_layers': DEFAULT_LAYERS,
            'n_heads': DEFAULT_HEADS,
            'regression_loss': 'mse',
        })

        return cfg, Tab2D


    def _split_data(self, X, y):
        """Split data into training and validation sets."""
        if hasattr(self, 'task') and self.task == 'classification':
            return make_stratified_dataset_split(X, y, seed=self.seed)
        else:
            # For regression, use random split
            val_indices = np.random.choice(range(len(X)), int(DEFAULT_VALIDATION_SPLIT * len(X)), replace=False).tolist()
            train_indices = [i for i in range(len(X)) if i not in val_indices]
            return X[train_indices], X[val_indices], y[train_indices], y[val_indices]

    def _train_ensemble(self, X_train, y_train, X_valid, y_valid, task, dim_output, n_classes=0, time_limit=None):
        """Train the ensemble of models."""

        cfg, Tab2D = self._create_config(task, dim_output, time_limit)
        rng = np.random.RandomState(cfg.seed)

        success = False
        while not (success and cfg.hyperparams["max_samples_support"] > 0 and cfg.hyperparams["max_samples_query"] > 0):
            try:
                self.trainers.clear()

                self.train_time = 0
                for _ in range(self.n_estimators):
                    if USE_HF:
                        if task == 'classification':
                            if self.hf_cls_model is not None:
                                model = Tab2D.from_pretrained(self.hf_cls_model, device=self.device)
                            elif self.hf_general_model is not None:
                                model = Tab2D.from_pretrained(self.hf_general_model, device=self.device)
                            else:
                                model = Tab2D.from_pretrained("autogluon/mitra-classifier", device=self.device)
                        elif task == 'regression':
                            if self.hf_reg_model is not None:
                                model = Tab2D.from_pretrained(self.hf_reg_model, device=self.device)
                            elif self.hf_general_model is not None:
                                model = Tab2D.from_pretrained(self.hf_general_model, device=self.device)
                            else:
                                model = Tab2D.from_pretrained("autogluon/mitra-regressor", device=self.device)
                    else:
                        model = Tab2D(
                            dim=cfg.hyperparams['dim'],
                            dim_output=dim_output,
                            n_layers=cfg.hyperparams['n_layers'],
                            n_heads=cfg.hyperparams['n_heads'],
                            task=task.upper(),
                            use_pretrained_weights=True,
                            path_to_weights=Path(self.state_dict),
                            device=self.device,
                        )
                    trainer = TrainerFinetune(cfg, model, n_classes=n_classes, device=self.device, rng=rng, verbose=self.verbose)

                    start_time = time.time()
                    trainer.train(X_train, y_train, X_valid, y_valid)
                    end_time = time.time()

                    self.trainers.append(trainer)
                    self.train_time += end_time - start_time

                    success = True

            except torch.cuda.OutOfMemoryError:
                if cfg.hyperparams["max_samples_support"] >= 2048:
                    cfg.hyperparams["max_samples_support"] = int(
                        cfg.hyperparams["max_samples_support"] // 2
                    )
                    print(f"Reducing max_samples_support from {cfg.hyperparams['max_samples_support'] * 2}"
                          f"to {cfg.hyperparams['max_samples_support']} due to OOM error.")
                else:
                    cfg.hyperparams["max_samples_support"] = int(
                        cfg.hyperparams["max_samples_support"] // 2
                    )
                    print(f"Reducing max_samples_support from {cfg.hyperparams['max_samples_support'] * 2}"
                          f"to {cfg.hyperparams['max_samples_support']} due to OOM error.")
                    cfg.hyperparams["max_samples_query"] = int(
                        cfg.hyperparams["max_samples_query"] // 2
                    )
                    print(f"Reducing max_samples_query from {cfg.hyperparams['max_samples_query'] * 2}"
                          f"to {cfg.hyperparams['max_samples_query']} due to OOM error.")

        if not success:
            raise RuntimeError(
                "Failed to train Mitra model after multiple attempts due to out of memory error."
            )

        return self


class MitraClassifier(MitraBase, ClassifierMixin):
    """Classifier implementation of Mitra model."""

    def __init__(self,
            model_type=DEFAULT_MODEL_TYPE,
            n_estimators=DEFAULT_ENSEMBLE,
            device=DEFAULT_DEVICE,
            fine_tune=DEFAULT_FINE_TUNE,
            fine_tune_steps=DEFAULT_FINE_TUNE_STEPS,
            metric=DEFAULT_CLS_METRIC,
            state_dict=None,
            patience=PATIENCE,
            lr=LR,
            warmup_steps=WARMUP_STEPS,
            shuffle_classes=SHUFFLE_CLASSES,
            shuffle_features=SHUFFLE_FEATURES,
            use_random_transforms=USE_RANDOM_TRANSFORMS,
            random_mirror_regression=RANDOM_MIRROR_REGRESSION,
            random_mirror_x=RANDOM_MIRROR_X,
            seed=SEED,
            verbose=True,
        ):
        """Initialize the classifier."""
        super().__init__(
            model_type,
            n_estimators,
            device,
            fine_tune,
            fine_tune_steps,
            metric,
            state_dict,
            patience=patience,
            lr=lr,
            warmup_steps=warmup_steps,
            shuffle_classes=shuffle_classes,
            shuffle_features=shuffle_features,
            use_random_transforms=use_random_transforms,
            random_mirror_regression=random_mirror_regression,
            random_mirror_x=random_mirror_x,
            seed=seed,
            verbose=verbose,
        )
        self.task = 'classification'

    def fit(self, X, y, X_val = None, y_val = None, time_limit = None):
        """
        Fit the ensemble of models.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
            Returns self
        """

        with mitra_deterministic_context():

            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(y, pd.Series):
                y = y.values

            self.X, self.y = X, y

            if X_val is not None and y_val is not None:
                if isinstance(X_val, pd.DataFrame):
                    X_val = X_val.values
                if isinstance(y_val, pd.Series):
                    y_val = y_val.values
                X_train, X_valid, y_train, y_valid = X, X_val, y, y_val
            else:
                X_train, X_valid, y_train, y_valid = self._split_data(X, y)

            return self._train_ensemble(X_train, y_train, X_valid, y_valid, self.task, DEFAULT_CLASSES, n_classes=DEFAULT_CLASSES, time_limit=time_limit)

    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes
        """

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples
            
        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples
        """

        with mitra_deterministic_context():

            if isinstance(X, pd.DataFrame):
                X = X.values

            preds = []
            for trainer in self.trainers:
                logits = trainer.predict(self.X, self.y, X)[...,:len(np.unique(self.y))] # Remove extra classes
                preds.append(np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)) # Softmax
            preds = sum(preds) / len(preds)  # Averaging ensemble predictions

            return preds


class MitraRegressor(MitraBase, RegressorMixin):
    """Regressor implementation of Mitra model."""

    def __init__(self,
            model_type=DEFAULT_MODEL_TYPE,
            n_estimators=DEFAULT_ENSEMBLE,
            device=DEFAULT_DEVICE,
            fine_tune=DEFAULT_FINE_TUNE,
            fine_tune_steps=DEFAULT_FINE_TUNE_STEPS,
            metric=DEFAULT_REG_METRIC,
            state_dict=None,
            patience=PATIENCE,
            lr=LR,
            warmup_steps=WARMUP_STEPS,
            shuffle_classes=SHUFFLE_CLASSES,
            shuffle_features=SHUFFLE_FEATURES,
            use_random_transforms=USE_RANDOM_TRANSFORMS,
            random_mirror_regression=RANDOM_MIRROR_REGRESSION,
            random_mirror_x=RANDOM_MIRROR_X,
            seed=SEED,
            verbose=True,
        ):
        """Initialize the regressor."""
        super().__init__(
            model_type,
            n_estimators,
            device,
            fine_tune,
            fine_tune_steps,
            metric,
            state_dict,
            patience=patience,
            lr=lr,
            warmup_steps=warmup_steps,
            shuffle_classes=shuffle_classes,
            shuffle_features=shuffle_features,
            use_random_transforms=use_random_transforms,
            random_mirror_regression=random_mirror_regression,
            random_mirror_x=random_mirror_x,
            seed=seed,
            verbose=verbose,
        )
        self.task = 'regression'

    def fit(self, X, y, X_val = None, y_val = None, time_limit = None):
        """
        Fit the ensemble of models.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
            Returns self
        """

        with mitra_deterministic_context():

            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(y, pd.Series):
                y = y.values

            self.X, self.y = X, y

            if X_val is not None and y_val is not None:
                if isinstance(X_val, pd.DataFrame):
                    X_val = X_val.values
                if isinstance(y_val, pd.Series):
                    y_val = y_val.values
                X_train, X_valid, y_train, y_valid = X, X_val, y, y_val
            else:
                X_train, X_valid, y_train, y_valid = self._split_data(X, y)

            return self._train_ensemble(X_train, y_train, X_valid, y_valid, self.task, 1, time_limit=time_limit)

    def predict(self, X):
        """
        Predict regression target for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values
        """

        with mitra_deterministic_context():

            if isinstance(X, pd.DataFrame):
                X = X.values
            
            preds = []
            for trainer in self.trainers:
                preds.append(trainer.predict(self.X, self.y, X))
        
            return sum(preds) / len(preds)  # Averaging ensemble predictions
    

@contextlib.contextmanager
def mitra_deterministic_context():
    """Context manager to set deterministic settings only for Mitra operations."""
    yield
