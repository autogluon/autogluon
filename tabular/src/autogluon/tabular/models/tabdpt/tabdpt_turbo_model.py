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
