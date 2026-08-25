"""AG-owned DenseLight trainer (preprocess + fit/predict). Architecture is LAMA; loop is AutoGluon."""

from __future__ import annotations

import logging
import math
import random
import time
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer

from autogluon.common.utils.random import get_numpy_seed
from autogluon.core.metrics import compute_metric

from ._denselight_net import DenseLightNet

if TYPE_CHECKING:
    from autogluon.core.metrics import Scorer

TaskType = Literal["regression", "binclass", "multiclass"]

logger = logging.getLogger(__name__)


def _auto_batch_size(n_train: int) -> int:
    if n_train < 2_800:
        return 32
    if n_train < 4_500:
        return 64
    if n_train < 6_400:
        return 128
    if n_train < 32_000:
        return 256
    if n_train < 108_000:
        return 512
    return 1024


class DenseLightImplementation:
    """Sklearn-style fit/predict wrapper around :class:`DenseLightNet`."""

    def __init__(self, early_stopping_metric: Scorer, num_classes: int | None = None, **config):
        self.config = config
        self.early_stopping_metric = early_stopping_metric
        # Authoritative class count from AbstractModel. Inferring it from the training labels
        # undersizes the output head whenever a class is absent from the split (small bagged
        # folds with a rare class), which silently returns too few predict_proba columns.
        self.num_classes = num_classes

        self.ord_enc_: OrdinalEncoder | None = None
        self.num_prep_: Pipeline | None = None
        self.cat_col_names_: list[Any] | None = None
        self.num_col_names_: list[Any] | None = None
        self.n_classes_: int | None = None
        self.task_type_: TaskType | None = None
        self.device_: torch.device | None = None
        self.model_: DenseLightNet | None = None
        self.best_val_score_: float | None = None
        self.y_mean_: float = 0.0
        self.y_std_: float = 1.0
        self.has_num_cols: bool = False
        self.has_cat_cols: bool = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        cat_col_names: list[Any],
        time_to_fit_in_seconds: float | None = None,
    ):
        start_time = time.time()
        if X_val is None or len(X_val) == 0:
            raise ValueError("Training without validation set is currently not implemented")

        seed: int | None = self.config.get("random_state", None)
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(get_numpy_seed(seed))
            random.seed(seed)
        if "n_threads" in self.config:
            torch.set_num_threads(self.config["n_threads"])

        problem_type = self.config["problem_type"]
        task_type: TaskType = "binclass" if problem_type == "binary" else problem_type
        n_train = len(X_train)
        device = torch.device(self.config["device"])
        self.task_type_ = task_type
        self.device_ = device
        self.cat_col_names_ = list(cat_col_names)
        self.num_col_names_ = [c for c in X_train.columns if c not in self.cat_col_names_]
        self.has_cat_cols = len(self.cat_col_names_) > 0
        self.has_num_cols = len(self.num_col_names_) > 0

        n_epochs = int(self.config.get("n_epochs", 100))
        patience = int(self.config.get("patience", 16))
        batch_size = self.config.get("batch_size", "auto")
        lr = float(self.config.get("lr", 1e-3))
        weight_decay = float(self.config.get("weight_decay", 1e-5))
        eval_batch_size = int(self.config.get("eval_batch_size", 1024))
        hidden_size = self.config.get("hidden_size", [512, 512])
        drop_rate = self.config.get("drop_rate", 0.1)
        use_bn = bool(self.config.get("use_bn", True))
        concat_input = bool(self.config.get("concat_input", True))
        dropout_first = bool(self.config.get("dropout_first", True))
        use_quantile = bool(self.config.get("use_quantile", True))
        gradient_clipping_norm = self.config.get("gradient_clipping_norm", 1.0)

        if batch_size == "auto":
            batch_size = _auto_batch_size(n_train)
        else:
            batch_size = int(batch_size)

        # --- Preprocess: cats ordinal (unknown/missing → cardinality), nums quantile+impute ---
        if self.has_cat_cols:
            self.ord_enc_ = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1,
            )
            self.ord_enc_.fit(X_train[self.cat_col_names_].astype("object"))
        if self.has_num_cols:
            steps = []
            if use_quantile:
                n_quantiles = max(min(n_train // 30, 1000), 10)
                steps.append(
                    (
                        "qt",
                        QuantileTransformer(
                            output_distribution="normal",
                            n_quantiles=n_quantiles,
                            subsample=1_000_000_000,
                            random_state=seed,
                        ),
                    )
                )
            steps.append(("imp", SimpleImputer(strategy="median")))
            self.num_prep_ = Pipeline(steps=steps)

        ds_parts: dict[str, dict[str, torch.Tensor]] = {}
        n_classes = 0
        for part, X, y in [("train", X_train, y_train), ("val", X_val, y_val)]:
            pieces: list[np.ndarray] = []
            if self.has_num_cols:
                x_num = X[self.num_col_names_].to_numpy(dtype=np.float32)
                if part == "train":
                    self.num_prep_.fit(x_num)
                pieces.append(self.num_prep_.transform(x_num).astype(np.float32, copy=False))
            if self.has_cat_cols:
                x_cat = self.ord_enc_.transform(X[self.cat_col_names_].astype("object"))
                # Map unknown/missing (-1) to one reserved code past known categories.
                for col_idx, cats in enumerate(self.ord_enc_.categories_):
                    unknown_code = float(len(cats))
                    mask = x_cat[:, col_idx] < 0
                    x_cat[mask, col_idx] = unknown_code
                pieces.append(x_cat.astype(np.float32, copy=False))
            if pieces:
                x_all = np.concatenate(pieces, axis=1)
            else:
                x_all = np.zeros((len(X), 1), dtype=np.float32)

            tensors: dict[str, torch.Tensor] = {
                "x": torch.as_tensor(x_all, dtype=torch.float32),
            }
            if task_type == "regression":
                tensors["y"] = torch.as_tensor(y.to_numpy(np.float32))
            else:
                tensors["y"] = torch.as_tensor(y.to_numpy(np.int64))
                if part == "train":
                    n_classes = max(n_classes, int(tensors["y"].max().item()) + 1)
            ds_parts[part] = tensors

        if self.num_classes is not None:
            n_classes = max(n_classes, int(self.num_classes))
        self.n_classes_ = n_classes
        n_in = int(ds_parts["train"]["x"].shape[1])
        n_out = 1 if task_type in ("regression", "binclass") else n_classes

        for part in ("train", "val"):
            for key in ds_parts[part]:
                ds_parts[part][key] = ds_parts[part][key].to(device)

        y_train_t = ds_parts["train"]["y"].clone()
        if task_type == "regression":
            self.y_mean_ = float(y_train_t.mean().item())
            self.y_std_ = float(y_train_t.std(correction=0).item())
            if self.y_std_ < 1e-12:
                self.y_std_ = 1.0
            y_train_t = (y_train_t - self.y_mean_) / self.y_std_

        model = DenseLightNet(
            n_in=n_in,
            n_out=n_out,
            hidden_size=hidden_size,
            drop_rate=drop_rate,
            use_bn=use_bn,
            concat_input=concat_input,
            dropout_first=dropout_first,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        if task_type == "regression":
            base_loss = nn.functional.mse_loss

            def loss_fn(logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
                return base_loss(logits.squeeze(-1), y_true)

        elif task_type == "binclass":

            def loss_fn(logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
                return nn.functional.binary_cross_entropy_with_logits(
                    logits.squeeze(-1),
                    y_true.float(),
                )

        else:

            def loss_fn(logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
                return nn.functional.cross_entropy(logits, y_true)

        @torch.inference_mode()
        def evaluate(part: str) -> float:
            model.eval()
            xs = ds_parts[part]["x"]
            ys = ds_parts[part]["y"]
            preds: list[torch.Tensor] = []
            for idx in torch.arange(len(ys), device=device).split(eval_batch_size):
                preds.append(model(xs[idx]))
            y_pred = torch.cat(preds, dim=0).float().cpu().numpy()
            y_true = ys.cpu().numpy()

            if task_type == "regression":
                y_pred = y_pred.reshape(-1) * self.y_std_ + self.y_mean_
                return compute_metric(
                    y=y_true,
                    metric=self.early_stopping_metric,
                    y_pred=y_pred,
                    y_pred_proba=None,
                    silent=True,
                )
            if task_type == "binclass":
                proba = 1.0 / (1.0 + np.exp(-y_pred.reshape(-1)))
                return compute_metric(
                    y=y_true,
                    metric=self.early_stopping_metric,
                    y_pred=(proba >= 0.5).astype(np.int64),
                    y_pred_proba=proba,
                    silent=True,
                )
            # multiclass
            proba = torch.softmax(torch.as_tensor(y_pred), dim=-1).numpy()
            return compute_metric(
                y=y_true,
                metric=self.early_stopping_metric,
                y_pred=proba.argmax(axis=1),
                y_pred_proba=proba,
                silent=True,
            )

        best = {"val": -math.inf, "epoch": -1}
        # state_dict, not parameters(): with use_bn the BatchNorm running statistics are buffers,
        # and restoring weights without them yields a model that was never the one evaluated.
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        remaining_patience = patience

        logger.log(15, f"DenseLight device={device.type} n_in={n_in} n_out={n_out} batch_size={batch_size}")
        for epoch in range(n_epochs):
            if epoch > 0 and time_to_fit_in_seconds is not None:
                pred_time_after_next_epoch = (epoch + 1) / epoch * (time.time() - start_time)
                if pred_time_after_next_epoch >= time_to_fit_in_seconds:
                    break

            model.train()
            perm = torch.randperm(n_train, device=device)
            batches = list(perm.split(batch_size))
            if len(batches) > 1 and len(batches[-1]) == 1:
                # BatchNorm1d raises on a single-row batch in train mode; fold it into the
                # previous batch rather than dropping the row.
                batches[-2] = torch.cat([batches[-2], batches[-1]])
                batches.pop()
            for batch_idx in batches:
                optimizer.zero_grad(set_to_none=True)
                logits = model(ds_parts["train"]["x"][batch_idx])
                loss = loss_fn(logits, y_train_t[batch_idx])
                loss.backward()
                if gradient_clipping_norm is not None and gradient_clipping_norm != "none":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clipping_norm))
                optimizer.step()

            val_score = evaluate("val")
            logger.log(15, f"DenseLight epoch={epoch} val={val_score:.6f}")
            if val_score > best["val"]:
                best = {"val": val_score, "epoch": epoch}
                remaining_patience = patience
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                remaining_patience -= 1
            if remaining_patience < 0:
                break

        model.load_state_dict(best_state)
        self.model_ = model
        self.best_val_score_ = best["val"]
        logger.log(15, f"DenseLight best={best}")

    def _transform_X(self, X: pd.DataFrame) -> torch.Tensor:
        pieces: list[np.ndarray] = []
        if self.has_num_cols:
            x_num = X[self.num_col_names_].to_numpy(dtype=np.float32)
            pieces.append(self.num_prep_.transform(x_num).astype(np.float32, copy=False))
        if self.has_cat_cols:
            x_cat = self.ord_enc_.transform(X[self.cat_col_names_].astype("object"))
            for col_idx, cats in enumerate(self.ord_enc_.categories_):
                unknown_code = float(len(cats))
                mask = x_cat[:, col_idx] < 0
                x_cat[mask, col_idx] = unknown_code
            pieces.append(x_cat.astype(np.float32, copy=False))
        if pieces:
            x_all = np.concatenate(pieces, axis=1)
        else:
            x_all = np.zeros((len(X), 1), dtype=np.float32)
        return torch.as_tensor(x_all, dtype=torch.float32, device=self.device_)

    def predict_raw(self, X: pd.DataFrame) -> torch.Tensor:
        assert self.model_ is not None and self.device_ is not None
        self.model_.eval()
        xs = self._transform_X(X)
        eval_batch_size = int(self.config.get("eval_batch_size", 1024))
        preds: list[torch.Tensor] = []
        with torch.inference_mode():
            for idx in torch.arange(xs.shape[0], device=self.device_).split(eval_batch_size):
                preds.append(self.model_(xs[idx]).float())
        y_pred = torch.cat(preds, dim=0)
        if self.task_type_ == "regression":
            return (y_pred.squeeze(-1) * self.y_std_ + self.y_mean_).cpu()
        if self.task_type_ == "binclass":
            # Return 2-class logits for a unified proba path.
            logits = y_pred.squeeze(-1)
            # stack as [logit_neg, logit_pos] with logit_neg=0 for sigmoid equivalence
            zeros = torch.zeros_like(logits)
            return torch.stack([zeros, logits], dim=1).cpu()
        return y_pred.cpu()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.predict_raw(X)
        if self.task_type_ == "regression":
            return raw.numpy()
        if self.task_type_ == "binclass":
            proba = torch.sigmoid(raw[:, 1]).numpy()
            return (proba >= 0.5).astype(np.int64)
        return raw.argmax(dim=-1).numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.predict_raw(X)
        if self.task_type_ == "binclass":
            return torch.sigmoid(raw[:, 1]).numpy()
        return torch.softmax(raw, dim=-1).numpy()
