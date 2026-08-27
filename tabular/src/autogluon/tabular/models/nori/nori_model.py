from __future__ import annotations

import contextlib
import logging
import os
from typing import TYPE_CHECKING

import numpy as np

from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.pretrained_weights import (
    PretrainedWeightsUnavailableError,
    fetch_allowed,
    unavailable_message,
)
from autogluon.tabular import __version__
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class NoriModel(AbstractTorchModel):
    """
    Nori is a tabular foundation model for regression via in-context learning.
    Given a few labeled context rows it predicts on query rows in a single forward
    pass, with no task-specific training. It is pretrained purely on synthetic data.

    Nori is regression-only; it does not support classification.

    Model size variants are selected with the ``model`` hyperparameter, which is
    forwarded to ``NoriRegressor``:

    - ``"nori"`` (default): the base ~6M-parameter checkpoint (``Synthefy/Nori``).
    - ``"nori-30m"``: the larger ~30M-parameter checkpoint (``Synthefy/Nori-30M``).

    For example::

        predictor.fit(..., hyperparameters={NoriModel: {"model": "nori-30m"}})

    Codebase: https://github.com/synthefy/synthefy-nori
    Model: https://huggingface.co/Synthefy/Nori (base), https://huggingface.co/Synthefy/Nori-30M (30M)
    License: Apache-2.0

    .. versionadded:: 1.6.0
    """

    gpu_strongly_recommended: bool = True  # in-context inference is 12-63x slower on CPU
    ag_key = "NORI"
    ag_name = "Nori"
    ag_priority = 40
    _supported_problem_types = ["regression"]

    _DEFAULT_MAX_BATCH_SIZE: int = 10000
    """Default prediction chunk size (``ag.max_batch_size``); also the query-batch
    bound assumed by the GPU memory estimate."""

    _INTERNAL_MAX_FEATURES: int = 256
    """Width the model actually sees, however wide the input is.

    Nori's inference config (``reg_allordinal_poly10_adaptive_svd256.json``, the default for every
    problem type) runs ``HighDimFeatureSelector`` with ``svd_components=256`` and
    ``n_features_threshold=256``, so anything wider is SVD-projected to 256 components before the
    model. Its ``MaxFeatureSubsampler`` (500) never fires as a result. Measured VRAM confirms it:
    it climbs with input width up to 256 features and is flat from there to 5000.
    """

    _default_auxiliary_params_extra = {
        # Nori attends queries over the full context with no internal chunking:
        # measured predict-phase VRAM is ~5 GB at 10k rows x 10 features,
        # ~21 GB at 10k x 100, and ~58 GB at the 50k-row cap (100 features),
        # so the cap only fits on high-memory GPUs.
        "max_rows": 50000,
        # Keep current behaviour: NoriRegressor never holds the pretrained module, so
        # AutoGluon has always saved Nori without weights (~0.2 MB). Defaulting to True
        # would copy the 45 MB checkpoint into every model directory and every bagged
        # fold. Set `ag.save_pretrained_weights=True` for an artifact that needs no
        # checkpoint at inference.
        "save_pretrained_weights": False,
        # No feature cap: the model sees at most `_INTERNAL_MAX_FEATURES` columns whatever the
        # input width (see that attribute), so a wide fit costs no more memory than a 256-feature
        # one -- measured 8.8 GB at both 256 and 5000 features.
        "max_features": None,
        # Chunk prediction: an unchunked 50k-query predict against a 50k-row
        # context fails with a CUDA kernel-configuration error (after ~76 GB);
        # 10k-query chunks on the same context run fine.
        "max_batch_size": _DEFAULT_MAX_BATCH_SIZE,
    }
    minimum_num_gpus = 0.5
    _default_ag_args_ensemble_extra = {
        "fold_fitting_strategy": "sequential_local",
        "refit_folds": True,  # Better to refit the model for faster inference and similar quality as the bag.
    }
    """Set fold_fitting_strategy to sequential_local, as parallel folding crashes if model weights aren't pre-downloaded."""
    default_resources_physical_cores_only = True
    default_num_gpus = 1

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        try:
            from synthefy_nori import NoriRegressor
        except ImportError as err:
            logger.log(
                40,
                f"\tFailed to import synthefy_nori! To use the Nori model, "
                f"do: `pip install autogluon.tabular[nori]=={__version__}`.",
            )
            raise err

        device = self._resolve_fit_device(num_gpus=num_gpus)

        hyp = self._get_model_params()
        hyp.pop("device", None)  # device is set explicitly from the allocated resources

        X = self.preprocess(X, y=y)
        y = y.to_numpy()
        # Only take over checkpoint resolution when the policy forbids fetching. Left alone,
        # NoriRegressor resolves its own checkpoint exactly as before, so the default path is
        # untouched -- including for callers who supply `model_path` themselves.
        if "model_path" not in hyp and not fetch_allowed(self.aux_params.fetch_pretrained_weights, stage="fit"):
            hyp["model_path"] = self._resolve_cached_checkpoint(model=hyp.get("model"))
        self.model = NoriRegressor(device=device, **hyp)
        self.model.fit(X=X, y=y)

    @classmethod
    def _resolve_cached_checkpoint(cls, model: str | None) -> str:
        """Resolve this variant's checkpoint from the local cache, or raise.

        Called only when fetching is disabled. Resolving the path ourselves and handing it to
        ``NoriRegressor`` as ``model_path`` is what keeps a disabled fetch from breaking an
        already-provisioned machine: a cached checkpoint still resolves, and only a real
        download is refused.
        """
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        from synthefy_nori.hf import (
            DEFAULT_CHECKPOINT_FILENAME,
            DEFAULT_MODEL_REPO_ID,
            resolve_model_repo,
        )

        repo_id = resolve_model_repo(model) if model is not None else DEFAULT_MODEL_REPO_ID
        try:
            return hf_hub_download(repo_id=repo_id, filename=DEFAULT_CHECKPOINT_FILENAME, local_files_only=True)
        except LocalEntryNotFoundError:
            raise PretrainedWeightsUnavailableError(
                unavailable_message(
                    model_name=cls.__name__, stage="fit", location=f"{repo_id}/{DEFAULT_CHECKPOINT_FILENAME}"
                )
            ) from None

    pretrained_weights_file_name = "nori_checkpoint.pt"
    """Filename of the embedded checkpoint under ``ag.save_pretrained_weights=True``."""

    def save(self, path: str | None = None, verbose: bool = True) -> str:
        """Save the fitted estimator, embedding the checkpoint only when asked.

        ``NoriRegressor`` never holds the pretrained module -- it builds its predictor lazily on
        the first ``predict`` and resolves the checkpoint then -- so the *unflagged* behaviour
        already matches ``ag.save_pretrained_weights=False``, and inference depends on the
        checkpoint still being resolvable. Honoring ``True`` therefore means copying the
        checkpoint into the artifact and pointing the estimator at that copy, which is what makes
        the artifact self-contained.
        """
        if not self.is_fit() or not self.aux_params.save_pretrained_weights:
            return super().save(path=path, verbose=verbose)

        import shutil

        path = path if path is not None else self.path
        os.makedirs(path, exist_ok=True)
        embedded = os.path.join(path, self.pretrained_weights_file_name)
        if not os.path.exists(embedded):
            source = self._resolve_cached_checkpoint(model=self._get_model_params().get("model"))
            shutil.copyfile(source, embedded)

        # Point the estimator at the copy, relative to the model directory so the artifact stays
        # movable; `load` resolves it back to an absolute path.
        previous = getattr(self.model, "model_path", None)
        self.model.model_path = self.pretrained_weights_file_name
        try:
            return super().save(path=path, verbose=verbose)
        finally:
            self.model.model_path = previous

    @classmethod
    def load(cls, path: str, reset_paths: bool = True, verbose: bool = True):
        """Load the pickle, pointing the estimator at an embedded checkpoint if one is present."""
        model = super().load(path=path, reset_paths=reset_paths, verbose=verbose)
        embedded = os.path.join(path, cls.pretrained_weights_file_name)
        if model.model is not None and os.path.exists(embedded):
            model.model.model_path = embedded
        return model

    def _predict_proba(self, X, **kwargs) -> np.ndarray:
        # Nori builds its predictor lazily on first predict and resolves the checkpoint then, so
        # unlike the other foundation models the inference-time fetch is not covered by gating
        # `_fit`. The load-stage policy governs here.
        with self._inference_fetch_policy():
            return self._predict_proba_inner(X, **kwargs)

    def _inference_fetch_policy(self):
        if fetch_allowed(self.aux_params.fetch_pretrained_weights, stage="load"):
            return contextlib.nullcontext()
        from ._weight_fetch import local_weights_only

        return local_weights_only(stage="load", model_name=type(self).__name__)

    def _predict_proba_inner(self, X, **kwargs) -> np.ndarray:
        X = self.preprocess(X, **kwargs)
        # Nori is regression-only: `predict` returns point estimates directly.
        return self.model.predict(X)

    def _preprocess(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Nori requires a fully numeric array as input; cast to float32 (the dtype
        Nori coerces to internally) with NaN preserved (handled natively)."""
        X = super()._preprocess(X, **kwargs)
        X = self._label_encode_categoricals(X)
        return np.asarray(X.to_numpy(), dtype=np.float32)

    def get_device(self) -> str:
        # NoriRegressor's device may be None (auto) or a torch.device; normalize to str.
        device = self.model.device
        if device is None:
            return "cpu"
        return device if isinstance(device, str) else device.type

    def _set_device(self, device: str):
        # Nori builds its inner predictor lazily on first predict, reading the device
        # off the regressor. Update the device and drop any cached predictor so the
        # next predict rebuilds on the new device (e.g. GPU -> CPU on load/save).
        self.model.device = device
        self.model._predictor = None

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,
        **kwargs,
    ) -> int:
        """CPU memory estimate: a ~3 GB process baseline (torch + model + inference
        buffers) plus the dataset footprint.

        Calibrated on measured fit+predict RSS (2.3-3.1 GB across 10k-50k rows,
        10-1000 features): the baseline dominates and the per-cell term is small,
        as Nori's activations live on the GPU.
        """
        baseline_mem_est = 3e9
        dataset_mem_est = 5 * get_approximate_df_mem_usage(X).sum()
        return int(baseline_mem_est + dataset_mem_est)

    @classmethod
    def _estimate_gpu_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,
        **kwargs,
    ) -> int:
        """Peak VRAM (reserved + CUDA context) across fit and prediction.

        Nori holds the whole labeled context on the device and attends over it for
        every query, so memory is driven by context *cells* (rows x features) rather
        than by rows alone: ~32 KB/cell up to ~1M cells, then ~8 KB/cell. The query
        batch adds ~0.5 MB/row, bounded by ``ag.max_batch_size`` chunking; as this
        bounds *fit* memory, where predictions are on held-out folds of the training
        data, the batch is also bounded by ``n_train``.

        This is substantial for a small model: measured peaks reach 95 GB on a
        24k-row x 387-feature task, so the estimate matters for scheduling.
        Calibrated on 30 real regression tasks (100 to 45k rows, 5 to 1024
        features): 1.0-2.2x of measured, no underestimates. The feature count is clamped to
        `_INTERNAL_MAX_FEATURES`, since wider inputs are projected down before the model.
        """
        n_train, n_features = X.shape
        max_batch_size = (hyperparameters or {}).get("ag.max_batch_size")
        if not isinstance(max_batch_size, int):
            max_batch_size = cls._DEFAULT_MAX_BATCH_SIZE
        n_test = min(max_batch_size, n_train)

        # Clamp to the width the model actually sees: beyond `_INTERNAL_MAX_FEATURES` the input is
        # SVD-projected, so scaling with the raw width overestimates badly (6.9x measured at 5000
        # features). Clamped, the estimate holds at 1.0-1.1x of measured from 100 to 5000 features.
        n_features = min(n_features, cls._INTERNAL_MAX_FEATURES)

        n_cells = n_train * n_features
        cell_saturation = 1e6
        return int(
            1.0e9  # CUDA context + model weights floor
            + 32e3 * min(n_cells, cell_saturation)
            + 8e3 * max(n_cells - cell_saturation, 0)
            + 0.5e6 * n_test
        )
