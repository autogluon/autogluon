"""Mitra v2 training and inference without process-global monkey patches."""

from __future__ import annotations

import gc
import warnings

import numpy as np
import torch

from ..data.dataset_finetune import DatasetFinetune
from .prediction_metrics import PredictionMetricsTracker
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


def _is_out_of_memory(error):
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


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
        x_train = np.asarray(x_train, dtype=np.float32)
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
        x_support = rng.standard_normal((n_support, features)).astype(np.float32)
        x_query = rng.standard_normal((n_query, features)).astype(np.float32)
        label_dtype = np.int64 if self.cfg.task == "classification" else np.float32
        # Use the released dataset and collator path, including its support
        # permutation and DataLoader RNG draw.
        dataset = DatasetFinetune(
            self.cfg,
            x_support=x_support,
            y_support=np.zeros(n_support, dtype=label_dtype),
            x_query=x_query,
            y_query=None,
            max_samples_support=n_support,
            max_samples_query=n_query,
            rng=rng,
        )
        batch = next(iter(self.make_loader(dataset, training=False)))
        was_training = self.model.training
        self.model.train()
        logits = None
        try:
            with torch.enable_grad():
                with torch.autocast(device_type="cuda", dtype=getattr(torch, self.cfg.hyperparams["precision"])):
                    logits = self._forward(batch)
                logits.float().mean().backward()
        finally:
            self.model.train(was_training)
            logits = None
            batch = None
            self.optimizer.zero_grad(set_to_none=True)
            for parameter in self.model.parameters():
                parameter.grad = None
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _forward(self, batch):
        x_support = batch["x_support"].to(self.device, non_blocking=True)
        y_support = batch["y_support"].to(self.device, non_blocking=True)
        x_query = batch["x_query"].to(self.device, non_blocking=True)
        padding_features = batch["padding_features"].to(self.device, non_blocking=True)
        padding_obs_support = batch["padding_obs_support"].to(self.device, non_blocking=True)
        padding_obs_query = batch["padding_obs_query"].to(self.device, non_blocking=True)
        if self.cfg.task == "regression" and self.cfg.hyperparams["regression_loss"] == "cross_entropy":
            y_support = torch.bucketize(y_support, self.bins) - 1
            y_support = torch.clamp(y_support, 0, self.cfg.hyperparams["dim_output"] - 1).to(torch.int64)
        return self.model(x_support, y_support, x_query, padding_features, padding_obs_support, padding_obs_query)

    def _predict_transformed(self, x_support, y_support, x_query, support_cap, query_chunk):
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
        dataset = PredictionDataset(**dataset_kwargs, balance_binary=self.cfg.hyperparams["balance_binary_support"])
        loader = self.make_loader(dataset, training=False)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in loader:
                with torch.autocast(
                    device_type=torch.device(self.device).type, dtype=getattr(torch, self.cfg.hyperparams["precision"])
                ):
                    logits = self._forward(batch)
                logits = logits[0].float()
                # Decode on the CPU and release GPU logits before the next batch.
                logits = logits.cpu()
                if self.cfg.task == "regression" and self.cfg.hyperparams["regression_loss"] == "cross_entropy":
                    logits = self._decode_regression(logits)
                logits = self.preprocessor.inverse_transform_y(logits.cpu().numpy())
                predictions.append(logits)
        return np.concatenate(predictions, axis=0)

    def evaluate(self, x_support, y_support, x_query, y_query):
        if (
            self._evaluation_cache is None
            or self._evaluation_cache[0] is not x_support
            or self._evaluation_cache[1] is not x_query
        ):
            self._evaluation_cache = (
                x_support,
                x_query,
                (
                    self.preprocessor.transform_X(x_support),
                    self.preprocessor.transform_y(y_support),
                    self.preprocessor.transform_X(x_query),
                    np.asarray(y_query),
                ),
            )
        query_floor = self.cfg.hyperparams["max_samples_query"]
        query_chunk = max(query_floor, min(self.cfg.hyperparams["eval_max_samples_query"], len(x_query)))
        while True:
            try:
                metrics = self._evaluate_transformed(*self._evaluation_cache[2], query_chunk=query_chunk)
                break
            except RuntimeError as error:
                if not _is_out_of_memory(error):
                    raise
                if query_chunk <= query_floor:
                    raise
                query_chunk = max(query_floor, query_chunk // 2)
            gc.collect()
            torch.cuda.empty_cache()
        objective = metrics.metrics[self.metric]
        metrics.loss = -objective if self.metric in ("accuracy", "auc", "r2", "f1") else objective
        return metrics

    def _evaluate_transformed(self, x_support, y_support, x_query, y_query, *, query_chunk):
        self.model.eval()
        dataset = DatasetFinetune(
            self.cfg,
            x_support=x_support,
            y_support=y_support,
            x_query=x_query,
            y_query=y_query,
            max_samples_support=self.cfg.hyperparams["max_samples_support"],
            max_samples_query=query_chunk,
            rng=self.rng,
        )
        loader = self.make_loader(dataset, training=False)
        tracker = PredictionMetricsTracker(task=self.cfg.task, preprocessor=self.preprocessor)
        with torch.inference_mode():
            for batch in loader:
                with torch.autocast(
                    device_type=torch.device(self.device).type, dtype=getattr(torch, self.cfg.hyperparams["precision"])
                ):
                    logits = self._forward(batch)
                query_y = batch["y_query"].to(self.device, non_blocking=True)
                if self.cfg.task == "regression" and self.cfg.hyperparams["regression_loss"] == "cross_entropy":
                    logits = self._decode_regression(logits)
                tracker.update(logits.float(), query_y, train=False)
        return tracker.get_metrics()

    def predict(self, x_support, y_support, x_query):
        transformed = (
            self.preprocessor.transform_X(x_support),
            self.preprocessor.transform_y(y_support),
            self.preprocessor.transform_X(x_query),
        )
        support_cap = self.cfg.hyperparams["predict_max_samples_support"]
        support_floor = min(support_cap, self.cfg.hyperparams["predict_min_samples_support"])
        stock_query_chunk = self.cfg.hyperparams["max_samples_query"]
        initial_query_chunk = max(stock_query_chunk, self.cfg.hyperparams["predict_max_samples_query"])
        query_chunk = initial_query_chunk
        while True:
            query_floor = min(
                query_chunk,
                self.cfg.hyperparams["predict_min_samples_query"]
                if len(x_support) <= support_cap
                else stock_query_chunk,
            )
            try:
                return self._predict_transformed(*transformed, support_cap, query_chunk)
            except RuntimeError as error:
                if not _is_out_of_memory(error):
                    raise
                if query_chunk > query_floor:
                    query_chunk = max(query_floor, query_chunk // 2)
                elif support_cap > support_floor:
                    next_cap = support_cap // 2
                    if support_cap > 32768 and next_cap < 32768:
                        next_cap = 32768
                    support_cap = max(support_floor, next_cap)
                    query_chunk = initial_query_chunk
                else:
                    raise
            # Release the failed forward before allocating the next attempt.
            gc.collect()
            torch.cuda.empty_cache()
