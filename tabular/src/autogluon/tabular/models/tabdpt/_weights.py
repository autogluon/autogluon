"""Detach and rebuild TabDPT's pretrained module for ``ag.save_pretrained_weights=False``.

``TabDPTEstimator`` loads its safetensors checkpoint inline in ``__init__`` and keeps the resulting
``nn.Module`` on ``.model``; there is no reload entry point to call, so rebuilding reproduces those
few lines. Two constructor arguments needed for the rebuild (``clip_sigma``, ``use_flash``) are not
kept on the estimator, only on the module itself, so they are recorded before it is dropped.
"""

from __future__ import annotations

from typing import Any

#: Attribute on the estimator holding what the rebuild needs. Pickled with the model.
REBUILD_ATTR = "_ag_pretrained_rebuild"


def capture_rebuild_args(estimator: Any) -> None:
    """Record what ``rebuild_pretrained`` will need, before the module is dropped."""
    module = estimator.model
    setattr(
        estimator,
        REBUILD_ATTR,
        {
            "clip_sigma": getattr(module, "clip_sigma", 8.0),
            "use_flash": getattr(module, "use_flash", getattr(estimator, "use_flash", True)),
        },
    )


def rebuild_pretrained(estimator: Any, checkpoint_path: str) -> None:
    """Reload the pretrained module from ``checkpoint_path`` onto ``estimator.model``."""
    import json

    from omegaconf import OmegaConf
    from safetensors import safe_open
    from tabdpt.model import TabDPTModel as _TabDPTNet

    args = getattr(estimator, REBUILD_ATTR, {}) or {}
    device = estimator.device

    with safe_open(checkpoint_path, framework="pt", device=device) as f:
        cfg = OmegaConf.create(json.loads(f.metadata()["cfg"]))
        model_state = {k: f.get_tensor(k) for k in f.keys()}

    cfg.env.device = device
    estimator.path = checkpoint_path
    estimator.model = _TabDPTNet.load(
        model_state=model_state,
        config=cfg,
        use_flash=args.get("use_flash", True),
        clip_sigma=args.get("clip_sigma", 8.0),
    )
    estimator.model.eval()
