"""Mitra v2 training and inference without process-global monkey patches."""

from __future__ import annotations

import gc
import warnings

import numpy as np
import torch

from ..data.dataset_finetune import DatasetFinetune
from .prediction_metrics import PredictionMetrics
from .trainer_finetune import TrainerFinetune


class DeviceCheckpoint:
    """Keep an independent copy of the best weights on the training device."""

    def reset(self, model):
        self.best_loss = np.inf
        self.best_model = {name: value.detach().clone() for name, value in model.state_dict().items()}

    def __call__(self, model, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            with torch.no_grad():
                for name, value in model.state_dict().items():
                    self.best_model[name].copy_(value)

    def set_to_best(self, model):
        model.load_state_dict(self.best_model)


class PredictionDataset(DatasetFinetune):
    """Balance capped binary support; reuse a full support draw across query chunks."""

    def __init__(self, *args, balance_binary=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.balance_binary = balance_binary
        self._fixed_support = None
        if self.support_size == self.n_samples_support:
            self._fixed_support = self._draw_support()

    def _draw_support(self):
        if self.balance_binary and self.support_size < self.n_samples_support:
            classes, counts = np.unique(self.y_support, return_counts=True)
            if len(classes) == 2:
                remaining = self.support_size
                indices = []
                for i, cls in enumerate(classes[np.argsort(counts)]):
                    candidates = np.flatnonzero(self.y_support == cls)
                    count = min(len(candidates), remaining // (2 - i))
                    indices.append(
                        candidates
                        if count == len(candidates)
                        else self.rng.choice(candidates, size=count, replace=False)
                    )
                    remaining -= count
                return np.concatenate(indices)
        return self.rng.choice(self.n_samples_support, size=self.support_size, replace=False)

    def __getitem__(self, index):
        support = self._fixed_support if self._fixed_support is not None else self._draw_support()
        return {
            "x_support": torch.as_tensor(self.x_support[support]),
            "y_support": torch.as_tensor(self.y_support[support]),
            "x_query": torch.as_tensor(self.x_queries[index]),
            "y_query": torch.as_tensor(self.y_queries[index]),
        }


class TrainerFinetuneV2(TrainerFinetune):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.cfg.hyperparams["fast_loop"]:
            self.checkpoint = DeviceCheckpoint()
            self.checkpoint.reset(self.model)
        self.early_stopping.metric = "roc_auc" if self.metric == "auc" else self.metric
        self._evaluation_cache = None

    def _decode_regression(self, logits):
        logits = logits.float()
        if not torch.isfinite(logits).all():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        centers = (self.bins[:-1] + self.bin_width / 2).to(logits.device)
        return (logits.softmax(dim=-1) * centers).sum(dim=-1)

    def train(self, x_train, y_train, x_val, y_val):
        if not self.cfg.hyperparams["fast_loop"]:
            return super().train(x_train, y_train, x_val, y_val)
        # The release uses SDPA in the training loop and the construction-time
        # attention backend for inference. These flags belong to this network.
        attention = [
            (module, module.use_flash_attn) for module in self.model.modules() if hasattr(module, "use_flash_attn")
        ]
        for module, _ in attention:
            module.use_flash_attn = False
        self._evaluation_cache = None
        try:
            if str(self.device).startswith("cuda") and self.cfg.hyperparams["max_epochs"] > 0:
                self._memory_preflight(x_train)
            return super().train(x_train, y_train, x_val, y_val)
        finally:
            self._evaluation_cache = None
            for module, original in attention:
                module.use_flash_attn = original

    def _memory_preflight(self, x_train):
        """Check that a training step fits before paying for full validation."""
        if len(x_train) < 2 or x_train.shape[1] == 0:
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            features = int((np.nanmax(x_train, axis=0) != np.nanmin(x_train, axis=0)).sum())
        if features == 0:
            return
        support_pool = int(np.ceil(0.8 * len(x_train)))
        n_support = min(support_pool, self.cfg.hyperparams["max_samples_support"])
        n_query = max(1, min(len(x_train) - support_pool, self.cfg.hyperparams["max_samples_query"]))
        rng = np.random.RandomState(0)
        label_dtype = torch.int64 if self.cfg.task == "classification" else torch.float32
        batch = {
            "x_support": torch.as_tensor(
                rng.standard_normal((1, n_support, features)).astype(np.float32), device=self.device
            ),
            "y_support": torch.zeros((1, n_support), device=self.device, dtype=label_dtype),
            "x_query": torch.as_tensor(
                rng.standard_normal((1, n_query, features)).astype(np.float32), device=self.device
            ),
            "padding_features": torch.zeros((1, features), device=self.device, dtype=torch.bool),
            "padding_obs_support": torch.zeros((1, n_support), device=self.device, dtype=torch.bool),
            "padding_obs_query": torch.zeros((1, n_query), device=self.device, dtype=torch.bool),
        }
        was_training = self.model.training
        self.model.train()
        try:
            with torch.autocast(device_type="cuda", dtype=getattr(torch, self.cfg.hyperparams["precision"])):
                logits = self._forward(batch)
            logits.float().mean().backward()
        finally:
            self.model.train(was_training)
            self.optimizer.zero_grad(set_to_none=True)

    def _forward(self, batch):
        tensors = {name: value.to(self.device, non_blocking=True) for name, value in batch.items()}
        support_y = tensors["y_support"]
        if self.cfg.task == "regression" and self.cfg.hyperparams["regression_loss"] == "cross_entropy":
            support_y = (
                (torch.bucketize(support_y, self.bins) - 1).clamp(0, self.cfg.hyperparams["dim_output"] - 1).long()
            )
        return self.model(
            tensors["x_support"],
            support_y,
            tensors["x_query"],
            tensors["padding_features"],
            tensors["padding_obs_support"],
            tensors["padding_obs_query"],
        )

    def _predict_transformed(self, x_support, y_support, x_query, support_cap, query_chunk, *, prediction):
        dataset_kwargs = dict(
            cfg=self.cfg,
            x_support=x_support,
            y_support=y_support,
            x_query=x_query,
            y_query=None,
            max_samples_support=support_cap,
            max_samples_query=query_chunk,
            rng=self.rng,
        )
        if prediction and self.cfg.hyperparams["fast_predict"]:
            dataset = PredictionDataset(
                **dataset_kwargs, balance_binary=self.cfg.hyperparams["balance_binary_support"]
            )
        else:
            dataset = DatasetFinetune(**dataset_kwargs)
        loader = self.make_loader(dataset, training=False)
        self.model.eval()
        predictions = []
        with torch.inference_mode():
            for batch in loader:
                with torch.autocast(
                    device_type=torch.device(self.device).type, dtype=getattr(torch, self.cfg.hyperparams["precision"])
                ):
                    logits = self._forward(batch)
                if self.cfg.task == "regression" and self.cfg.hyperparams["regression_loss"] == "cross_entropy":
                    if prediction:
                        # The released predictor decodes float32 logits on CPU.
                        # Keep its reduction order; validation decodes on the model device.
                        logits = logits.float().cpu()
                    logits = self._decode_regression(logits)
                predictions.append(logits[0].float().cpu().numpy())
        return self.preprocessor.inverse_transform_y(np.concatenate(predictions, axis=0))

    def evaluate(self, x_support, y_support, x_query, y_query):
        if not self.cfg.hyperparams["fast_loop"]:
            return super().evaluate(x_support, y_support, x_query, y_query)
        if self._evaluation_cache is None:
            self._evaluation_cache = (
                self.preprocessor.transform_X(x_support),
                self.preprocessor.transform_y(y_support),
                self.preprocessor.transform_X(x_query),
            )
        query_chunk = self.cfg.hyperparams["eval_max_samples_query"]
        query_floor = min(query_chunk, 1024)
        rng_state = self.rng.get_state()
        while True:
            try:
                predictions = self._predict_transformed(
                    *self._evaluation_cache,
                    self.cfg.hyperparams["max_samples_support"],
                    query_chunk,
                    prediction=False,
                )
                break
            except torch.cuda.OutOfMemoryError:
                self.rng.set_state(rng_state)
                if query_chunk <= query_floor:
                    raise
                query_chunk = max(query_floor, query_chunk // 2)
                self.cfg.hyperparams["eval_max_samples_query"] = query_chunk
                gc.collect()
                torch.cuda.empty_cache()
        metrics = PredictionMetrics.from_prediction(predictions, np.asarray(y_query), self.cfg.task)
        objective = metrics.metrics[self.metric]
        metrics.loss = -objective if self.metric in ("accuracy", "auc", "r2", "f1") else objective
        return metrics

    def predict(self, x_support, y_support, x_query):
        transformed = (
            self.preprocessor.transform_X(x_support),
            self.preprocessor.transform_y(y_support),
            self.preprocessor.transform_X(x_query),
        )
        support_cap = self.cfg.hyperparams["predict_max_samples_support"]
        support_floor = min(support_cap, self.cfg.hyperparams["predict_min_samples_support"])
        query_chunk = self.cfg.hyperparams["predict_max_samples_query"]
        rng_state = self.rng.get_state()
        retry_without_traceback = False
        while True:
            query_floor = min(
                query_chunk,
                self.cfg.hyperparams["predict_min_samples_query"] if len(x_support) <= support_cap else 1024,
            )
            try:
                return self._predict_transformed(*transformed, support_cap, query_chunk, prediction=True)
            except torch.cuda.OutOfMemoryError:
                self.rng.set_state(rng_state)
                if query_chunk > query_floor:
                    query_chunk = max(query_floor, query_chunk // 2)
                elif support_cap > support_floor:
                    support_cap = max(support_floor, support_cap // 2)
                else:
                    if retry_without_traceback:
                        raise
                    retry_without_traceback = True
                if not retry_without_traceback:
                    gc.collect()
                    torch.cuda.empty_cache()
            if retry_without_traceback:
                # The failed forward's traceback retains its tensors inside
                # the exception handler. Release them before one final retry
                # with the same support, query chunk, and random state.
                gc.collect()
                torch.cuda.empty_cache()
