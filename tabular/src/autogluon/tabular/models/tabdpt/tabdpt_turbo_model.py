from __future__ import annotations

from typing import ClassVar

from .tabdpt_model import TabDPTModel


class TabDPTTurboModel(TabDPTModel):
    """TabDPT-Turbo (TabDPT v1.2).

    Accelerates fitting and inference by ~120x on average versus v1.1 while improving
    predictive performance, chiefly by defaulting to subsampled context reduction
    (instead of v1.1's FAISS retrieval) plus long-context support and updated weights.
    Exposes the v1.2 predict knobs (``n_ensembles`` / ``batch_size``).

    Paper: "TabDPT-Turbo" — https://openreview.net/pdf?id=Y00pwFyrHR
    Requires ``tabdpt>=1.2``.

    .. versionadded:: 1.6.0
    """

    ag_key = "TABDPT-TURBO"
    ag_name = "TabDPT-Turbo"

    _checkpoint_filename: ClassVar[str] = "tabdpt1_2.safetensors"
    _constructor_defaults: ClassVar[dict[str, object]] = {
        "normalizer": "standard",
        "missing_indicators": False,
        "clip_sigma": 8,  # v1.2 default (v1.1 uses 4)
        "feature_reduction": "pca",
        "context_reduction": "subsample",
        "faiss_metric": "l2",
    }
    _predict_param_names: ClassVar[dict[str, tuple[str, ...]]] = {
        "classifier": ("context_size", "n_ensembles", "batch_size", "permute_classes", "temperature"),
        "regressor": ("context_size", "n_ensembles", "batch_size"),
    }

    @classmethod
    def _estimate_gpu_memory_usage_static(
        cls,
        *,
        X,
        hyperparameters: dict | None = None,
        **kwargs,
    ) -> int:
        """Peak VRAM (reserved + CUDA context) across fit and prediction.

        TabDPT-Turbo's peak is a ~1.3 GB floor (CUDA context + weights), which
        dominates small datasets, plus a feature-independent row term (fixed-width
        encoding of a subsampled context, ~25 KB per train + prediction row).

        This bounds *fit* memory, where predictions are on held-out folds of the
        training data, so the prediction-row count is taken as ``n_train``.
        (Inference on a test set far larger than the training data can exceed this;
        that is inference-time memory, which AutoGluon's fit-time checks do not
        cover.) Calibrated on synthetic fit+predict measurements (1k-100k rows,
        10-1000 features, up to 200k prediction rows) plus all 136 real TabArena and
        BeyondArena tasks (100 to 1M rows): 1.0-1.9x of measured, no underestimates.
        """
        n_train = len(X)
        n_test = n_train
        return int(1.3e9 + 25e3 * (n_train + n_test))
