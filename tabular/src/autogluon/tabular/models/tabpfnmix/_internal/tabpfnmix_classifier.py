from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import torch

from .core.dataset_split import make_stratified_dataset_split
from .core.trainer_finetune import TrainerFinetune
from .models.foundation.foundation_transformer import FoundationTransformer


# FIXME: Delete this class, it isn't needed
# TODO: test_epoch might be better if it uses the for loop logic with n_ensembles during finetuning to better estimate val score
# TODO: To mitigate val overfitting, can fit multiple random seeds at same time and pick same epoch for all of them, track average performance on epoch.
# TODO: Test shuffling the data and see if it makes TabPFNv2 worse, same with TabForestPFN
class TabPFNMixClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_classes, cfg, split_val, model_path: str = None, weights_path: str | Path = None, stopping_metric=None, use_best_epoch: bool = True):
        if weights_path is not None:
            weights_path = str(Path(weights_path))

        if model_path is not None:
            model = FoundationTransformer.from_pretrained(model_path)
            assert model.task == cfg.task, f"The pretrained model '{model_path}' is for task {model.task}, but the problem type is for task {cfg.task}..."
        else:
            model = FoundationTransformer(
                n_features=cfg.hyperparams['n_features'],
                n_classes=cfg.hyperparams['n_classes'],
                dim=cfg.hyperparams['dim'],
                n_layers=cfg.hyperparams['n_layers'],
                n_heads=cfg.hyperparams['n_heads'],
                attn_dropout=cfg.hyperparams['attn_dropout'],
                y_as_float_embedding=cfg.hyperparams['y_as_float_embedding'],
                task=cfg.task,
            )
        if weights_path is not None:
            model.load_state_dict(torch.load(weights_path, weights_only=True))  # nosec B614

        self.split_val = split_val
        self.trainer = TrainerFinetune(cfg, model, n_classes=n_classes, stopping_metric=stopping_metric, use_best_epoch=use_best_epoch)
        super().__init__()

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, time_limit: float = None):
        # FIXME: Should X and y be preprocessed for inference efficiency? Yes.
        self.X_ = X  # FIXME: Optimize storage of X and y? Is this redundant? Is X and y saving done multiple times during pickle?
        self.y_ = y

        if X_val is not None and y_val is not None:
            X_train, X_valid, y_train, y_valid = X, X_val, y, y_val
        elif self.split_val:
            X_train, X_valid, y_train, y_valid = make_stratified_dataset_split(X, y)  # FIXME: Add random seed
        else:
            X_train, X_valid, y_train, y_valid = X, None, y, None
        self.trainer.train(x_train=X_train, y_train=y_train, x_val=X_valid, y_val=y_valid, time_limit=time_limit)

        return self

    # FIXME: Avoid preprocessing self.X_ and self.y_ each predict call
    def predict(self, X):
        logits = self.trainer.predict(self.X_, self.y_, X)
        return logits.argmax(axis=1)

    # FIXME: Avoid preprocessing self.X_ and self.y_ each predict_proba call
    def predict_proba(self, X):
        logits = self.trainer.predict(self.X_, self.y_, X)
        return np.exp(logits) / np.exp(logits).sum(axis=1)[:, None]