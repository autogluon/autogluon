"""Partially adapted from pytabkit's TabM implementation."""

from __future__ import annotations

import logging
import math
import random
import time
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import scipy
import torch
from autogluon.core.metrics import compute_metric
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer
from sklearn.utils.validation import check_is_fitted

from . import rtdl_num_embeddings, tabm_reference
from .tabm_reference import make_parameter_groups

if TYPE_CHECKING:
    from autogluon.core.metrics import Scorer

TaskType = Literal["regression", "binclass", "multiclass"]

logger = logging.getLogger(__name__)


def get_tabm_auto_batch_size(n_train: int) -> int:
    # by Yury Gorishniy, inferred from the choices in the TabM paper.
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


class RTDLQuantileTransformer(BaseEstimator, TransformerMixin):
    # adapted from pytabkit
    def __init__(
        self,
        noise=1e-5,
        random_state=None,
        n_quantiles=1000,
        subsample=1_000_000_000,
        output_distribution="normal",
    ):
        self.noise = noise
        self.random_state = random_state
        self.n_quantiles = n_quantiles
        self.subsample = subsample
        self.output_distribution = output_distribution

    def fit(self, X, y=None):
        # Calculate the number of quantiles based on data size
        n_quantiles = max(min(X.shape[0] // 30, self.n_quantiles), 10)

        # Initialize QuantileTransformer
        normalizer = QuantileTransformer(
            output_distribution=self.output_distribution,
            n_quantiles=n_quantiles,
            subsample=self.subsample,
            random_state=self.random_state,
        )

        # Add noise if required
        X_modified = self._add_noise(X) if self.noise > 0 else X

        # Fit the normalizer
        normalizer.fit(X_modified)
        # show that it's fitted
        self.normalizer_ = normalizer

        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        return self.normalizer_.transform(X)

    def _add_noise(self, X):
        return X + np.random.default_rng(self.random_state).normal(0.0, 1e-5, X.shape).astype(X.dtype)


class TabMOrdinalEncoder(BaseEstimator, TransformerMixin):
    # encodes missing and unknown values to a value one larger than the known values
    def __init__(self):
        # No fitted attributes here — only parameters
        pass

    def fit(self, X, y=None):
        X = pd.DataFrame(X)

        # Fit internal OrdinalEncoder with NaNs preserved for now
        self.encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
            encoded_missing_value=np.nan,
        )
        self.encoder_.fit(X)

        # Cardinalities = number of known categories per column
        self.cardinalities_ = [len(cats) for cats in self.encoder_.categories_]

        return self

    def transform(self, X):
        check_is_fitted(self, ["encoder_", "cardinalities_"])

        X = pd.DataFrame(X)
        X_enc = self.encoder_.transform(X)

        # Replace np.nan (unknown or missing) with cardinality value
        for col_idx, cardinality in enumerate(self.cardinalities_):
            mask = np.isnan(X_enc[:, col_idx])
            X_enc[mask, col_idx] = cardinality

        return X_enc.astype(int)

    def get_cardinalities(self):
        check_is_fitted(self, ["cardinalities_"])
        return self.cardinalities_


class TabMImplementation:
    def __init__(self, early_stopping_metric: Scorer, **config):
        self.config = config
        self.early_stopping_metric = early_stopping_metric

        self.ord_enc_ = None
        self.num_prep_ = None
        self.cat_col_names_ = None
        self.n_classes_ = None
        self.task_type_ = None
        self.device_ = None
        self.has_num_cols = None

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
            np.random.seed(seed)
            random.seed(seed)
        if "n_threads" in self.config:
            torch.set_num_threads(self.config["n_threads"])

        # -- Meta parameters
        problem_type = self.config["problem_type"]
        task_type: TaskType = "binclass" if problem_type == "binary" else problem_type
        n_train = len(X_train)
        n_classes = None
        device = self.config["device"]
        device = torch.device(device)
        self.task_type_ = task_type
        self.device_ = device
        self.cat_col_names_ = cat_col_names

        # -- Hyperparameters
        arch_type = self.config.get("arch_type", "tabm-mini")
        num_emb_type = self.config.get("num_emb_type", "pwl")
        n_epochs = self.config.get("n_epochs", 1_000_000_000)
        patience = self.config.get("patience", 16)
        batch_size = self.config.get("batch_size", "auto")
        compile_model = self.config.get("compile_model", False)
        lr = self.config.get("lr", 2e-3)
        d_embedding = self.config.get("d_embedding", 16)
        d_block = self.config.get("d_block", 512)
        dropout = self.config.get("dropout", 0.1)
        tabm_k = self.config.get("tabm_k", 32)
        allow_amp = self.config.get("allow_amp", False)
        n_blocks = self.config.get("n_blocks", "auto")
        num_emb_n_bins = self.config.get("num_emb_n_bins", 48)
        eval_batch_size = self.config.get("eval_batch_size", 1024)
        share_training_batches = self.config.get("share_training_batches", False)
        weight_decay = self.config.get("weight_decay", 3e-4)
        # this is the search space default but not the example default (which is 'none')
        gradient_clipping_norm = self.config.get("gradient_clipping_norm", 1.0)

        # -- Verify HPs
        num_emb_n_bins = min(num_emb_n_bins, n_train - 1)
        if n_train <= 2:
            num_emb_type = "none"  # there is no valid number of bins for piecewise linear embeddings
        if batch_size == "auto":
            batch_size = get_tabm_auto_batch_size(n_train=n_train)

        # -- Preprocessing
        ds_parts = dict()
        self.ord_enc_ = (
            TabMOrdinalEncoder()
        )  # Unique ordinal encoder -> replaces nan and missing values with the cardinality
        self.ord_enc_.fit(X_train[self.cat_col_names_])
        # TODO: fix transformer to be able to work with empty input data like the sklearn default
        self.num_prep_ = Pipeline(steps=[
            ("qt", RTDLQuantileTransformer(random_state=self.config.get("random_state", None))),
            ("imp", SimpleImputer(add_indicator=True)),
        ])
        self.has_num_cols = bool(set(X_train.columns) - set(cat_col_names))
        for part, X, y in [("train", X_train, y_train), ("val", X_val, y_val)]:
            tensors = dict()

            tensors["x_cat"] = torch.as_tensor(self.ord_enc_.transform(X[cat_col_names]), dtype=torch.long)

            if self.has_num_cols:
                x_cont_np = X.drop(columns=cat_col_names).to_numpy(dtype=np.float32)
                if part == "train":
                    self.num_prep_.fit(x_cont_np)
                tensors["x_cont"] = torch.as_tensor(self.num_prep_.transform(x_cont_np))
            else:
                tensors["x_cont"] = torch.empty((len(X), 0), dtype=torch.float32)

            if task_type == "regression":
                tensors["y"] = torch.as_tensor(y.to_numpy(np.float32))
                if part == "train":
                    n_classes = 0
            else:
                tensors["y"] = torch.as_tensor(y.to_numpy(np.int32), dtype=torch.long)
                if part == "train":
                    n_classes = tensors["y"].max().item() + 1

            ds_parts[part] = tensors

        part_names = ["train", "val"]
        cat_cardinalities = self.ord_enc_.get_cardinalities()
        self.n_classes_ = n_classes

        # filter out numerical columns with only a single value
        #  -> AG also does this already but preprocessing might create constant columns again
        x_cont_train = ds_parts["train"]["x_cont"]
        self.num_col_mask_ = ~torch.all(x_cont_train == x_cont_train[0:1, :], dim=0)
        for part in part_names:
            ds_parts[part]["x_cont"] = ds_parts[part]["x_cont"][:, self.num_col_mask_]
            # tensor infos are not correct anymore, but might not be used either
        for part in part_names:
            for tens_name in ds_parts[part]:
                ds_parts[part][tens_name] = ds_parts[part][tens_name].to(device)

        # update
        n_cont_features = ds_parts["train"]["x_cont"].shape[1]

        Y_train = ds_parts["train"]["y"].clone()
        if task_type == "regression":
            self.y_mean_ = ds_parts["train"]["y"].mean().item()
            self.y_std_ = ds_parts["train"]["y"].std(correction=0).item()

            Y_train = (Y_train - self.y_mean_) / (self.y_std_ + 1e-30)

        # the | operator joins dicts (like update() but not in-place)
        data = {
            part: dict(x_cont=ds_parts[part]["x_cont"], y=ds_parts[part]["y"])
            | (dict(x_cat=ds_parts[part]["x_cat"]) if ds_parts[part]["x_cat"].shape[1] > 0 else dict())
            for part in part_names
        }

        # adapted from https://github.com/yandex-research/tabm/blob/main/example.ipynb

        # Automatic mixed precision (AMP)
        # torch.float16 is implemented for completeness,
        # but it was not tested in the project,
        # so torch.bfloat16 is used by default.
        amp_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
            if torch.cuda.is_available()
            else None
        )
        # Changing False to True will result in faster training on compatible hardware.
        amp_enabled = allow_amp and amp_dtype is not None
        grad_scaler = torch.cuda.amp.GradScaler() if amp_dtype is torch.float16 else None  # type: ignore

        # fmt: off
        logger.log(15, f"Device:        {device.type.upper()}"
                    f"\nAMP:           {amp_enabled} (dtype: {amp_dtype})"
                    f"\ntorch.compile: {compile_model}",
                    )
        # fmt: on

        bins = (
            None
            if num_emb_type != "pwl" or n_cont_features == 0
            else rtdl_num_embeddings.compute_bins(data["train"]["x_cont"], n_bins=num_emb_n_bins)
        )

        model = tabm_reference.Model(
            n_num_features=n_cont_features,
            cat_cardinalities=cat_cardinalities,
            n_classes=n_classes if n_classes > 0 else None,
            backbone={
                "type": "MLP",
                "n_blocks": n_blocks if n_blocks != "auto" else (3 if bins is None else 2),
                "d_block": d_block,
                "dropout": dropout,
            },
            bins=bins,
            num_embeddings=(
                None
                if bins is None
                else {
                    "type": "PiecewiseLinearEmbeddings",
                    "d_embedding": d_embedding,
                    "activation": False,
                    "version": "B",
                }
            ),
            arch_type=arch_type,
            k=tabm_k,
            share_training_batches=share_training_batches,
        ).to(device)
        optimizer = torch.optim.AdamW(make_parameter_groups(model), lr=lr, weight_decay=weight_decay)

        if compile_model:
            # NOTE
            # `torch.compile` is intentionally called without the `mode` argument
            # (mode="reduce-overhead" caused issues during training with torch==2.0.1).
            model = torch.compile(model)
            evaluation_mode = torch.no_grad
        else:
            evaluation_mode = torch.inference_mode

        @torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)  # type: ignore[code]
        def apply_model(part: str, idx: torch.Tensor) -> torch.Tensor:
            return (
                model(
                    data[part]["x_cont"][idx],
                    data[part]["x_cat"][idx] if "x_cat" in data[part] else None,
                )
                .squeeze(-1)  # Remove the last dimension for regression tasks.
                .float()
            )

        # TODO: use BCELoss for binary classification
        base_loss_fn = torch.nn.functional.mse_loss if task_type == "regression" else torch.nn.functional.cross_entropy

        def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            # TabM produces k predictions per object. Each of them must be trained separately.
            # (regression)     y_pred.shape == (batch_size, k)
            # (classification) y_pred.shape == (batch_size, k, n_classes)
            k = y_pred.shape[1]
            return base_loss_fn(
                y_pred.flatten(0, 1),
                y_true.repeat_interleave(k) if model.share_training_batches else y_true,
            )

        @evaluation_mode()
        def evaluate(part: str) -> float:
            model.eval()

            # When using torch.compile, you may need to reduce the evaluation batch size.
            y_pred: np.ndarray = (
                torch.cat(
                    [
                        apply_model(part, idx)
                        for idx in torch.arange(len(data[part]["y"]), device=device).split(
                            eval_batch_size,
                        )
                    ],
                )
                .cpu()
                .numpy()
            )
            if task_type == "regression":
                # Transform the predictions back to the original label space.
                y_pred = y_pred * self.y_std_ + self.y_mean_

            # Compute the mean of the k predictions.
            average_logits = self.config.get("average_logits", False)
            if average_logits:
                y_pred = y_pred.mean(1)
            if task_type != "regression":
                # For classification, the mean must be computed in the probability space.
                y_pred = scipy.special.softmax(y_pred, axis=-1)
            if not average_logits:
                y_pred = y_pred.mean(1)

            return compute_metric(
                y=data[part]["y"].cpu().numpy(),
                metric=self.early_stopping_metric,
                y_pred=y_pred if task_type == "regression" else y_pred.argmax(1),
                y_pred_proba=y_pred[:, 1] if task_type == "binclass" else y_pred,
                silent=True,
            )

        math.ceil(n_train / batch_size)
        best = {
            "val": -math.inf,
            # 'test': -math.inf,
            "epoch": -1,
        }
        best_params = [p.clone() for p in model.parameters()]
        # Early stopping: the training stops when
        # there are more than `patience` consecutive bad updates.
        remaining_patience = patience

        try:
            if self.config.get("verbosity", 0) >= 1:
                from tqdm.std import tqdm
            else:
                tqdm = lambda arr, desc: arr
        except ImportError:
            tqdm = lambda arr, desc: arr

        logger.log(15, "-" * 88 + "\n")
        for epoch in range(n_epochs):
            # check time limit
            if epoch > 0 and time_to_fit_in_seconds is not None:
                pred_time_after_next_epoch = (epoch + 1) / epoch * (time.time() - start_time)
                if pred_time_after_next_epoch >= time_to_fit_in_seconds:
                    break

            batches = (
                torch.randperm(n_train, device=device).split(batch_size)
                if model.share_training_batches
                else [
                    x.transpose(0, 1).flatten()
                    for x in torch.rand((model.k, n_train), device=device).argsort(dim=1).split(batch_size, dim=1)
                ]
            )

            for batch_idx in tqdm(batches, desc=f"Epoch {epoch}"):
                model.train()
                optimizer.zero_grad()
                loss = loss_fn(apply_model("train", batch_idx), Y_train[batch_idx])

                # added from https://github.com/yandex-research/tabm/blob/main/bin/model.py
                if gradient_clipping_norm is not None and gradient_clipping_norm != "none":
                    if grad_scaler is not None:
                        grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad.clip_grad_norm_(
                        model.parameters(),
                        gradient_clipping_norm,
                    )

                if grad_scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    grad_scaler.scale(loss).backward()  # type: ignore
                    grad_scaler.step(optimizer)  # Ignores grad scaler might skip steps; should not break anything
                    grad_scaler.update()

            val_score = evaluate("val")
            logger.log(15, f"(val) {val_score:.4f}")

            if val_score > best["val"]:
                logger.log(15, "🌸 New best epoch! 🌸")
                # best = {'val': val_score, 'test': test_score, 'epoch': epoch}
                best = {"val": val_score, "epoch": epoch}
                remaining_patience = patience
                with torch.no_grad():
                    for bp, p in zip(best_params, model.parameters(), strict=False):
                        bp.copy_(p)
            else:
                remaining_patience -= 1

            if remaining_patience < 0:
                break

        logger.log(15, "\n\nResult:")
        logger.log(15, str(best))

        logger.log(15, "Restoring best model")
        with torch.no_grad():
            for bp, p in zip(best_params, model.parameters(), strict=False):
                p.copy_(bp)

        self.model_ = model

    def predict_raw(self, X: pd.DataFrame) -> torch.Tensor:
        self.model_.eval()

        tensors = dict()
        tensors["x_cat"] = torch.as_tensor(self.ord_enc_.transform(X[self.cat_col_names_]), dtype=torch.long).to(
            self.device_,
        )
        tensors["x_cont"] = torch.as_tensor(
            self.num_prep_.transform(X.drop(columns=X[self.cat_col_names_]).to_numpy(dtype=np.float32))
            if self.has_num_cols
            else np.empty((len(X), 0), dtype=np.float32),
        ).to(self.device_)

        tensors["x_cont"] = tensors["x_cont"][:, self.num_col_mask_]

        eval_batch_size = self.config.get("eval_batch_size", 1024)
        with torch.no_grad():
            y_pred: torch.Tensor = torch.cat(
                [
                    self.model_(
                        tensors["x_cont"][idx],
                        tensors["x_cat"][idx] if tensors["x_cat"].numel() != 0 else None,
                    )
                    .squeeze(-1)  # Remove the last dimension for regression tasks.
                    .float()
                    for idx in torch.arange(tensors["x_cont"].shape[0], device=self.device_).split(
                        eval_batch_size,
                    )
                ],
            )
        if self.task_type_ == "regression":
            # Transform the predictions back to the original label space.
            y_pred = y_pred * self.y_std_ + self.y_mean_
            y_pred = y_pred.mean(1)
            # y_pred = y_pred.unsqueeze(-1)  # add extra "features" dimension
        else:
            average_logits = self.config.get("average_logits", False)
            if average_logits:
                y_pred = y_pred.mean(1)
            else:
                # For classification, the mean must be computed in the probability space.
                y_pred = torch.log(torch.softmax(y_pred, dim=-1).mean(1) + 1e-30)

        return y_pred.cpu()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y_pred = self.predict_raw(X)
        if self.task_type_ == "regression":
            return y_pred.numpy()
        return y_pred.argmax(dim=-1).numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probas = torch.softmax(self.predict_raw(X), dim=-1).numpy()
        if probas.shape[1] == 2:
            probas = probas[:, 1]
        return probas
