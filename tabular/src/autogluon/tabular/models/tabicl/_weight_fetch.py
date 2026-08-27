"""Enforcement of ``ag.fetch_pretrained_weights`` for TabICL at load time.

At *fit* time the policy is simply passed to the estimator's ``allow_auto_download`` parameter.
Load is different: tabicl's ``__setstate__`` reloads the checkpoint during unpickling, so the fetch
happens before the AutoGluon model object -- and therefore its ``aux_params`` -- exists. The only
policy source available at that moment is the environment, which is also the right scope: an
inference-time policy belongs to the deployment rather than to the artifact.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

from autogluon.common.utils.pretrained_weights import (
    PretrainedWeightsUnavailableError,
    fetch_allowed,
    unavailable_message,
)

__all__ = ["weight_fetch_policy"]

_PATCH_TARGETS = ("tabicl._sklearn.classifier", "tabicl._sklearn.regressor")


@contextlib.contextmanager
def weight_fetch_policy(*, stage: str, model_name: str) -> Iterator[None]:
    """Block remote checkpoint fetches for the duration when the environment forbids them.

    ``aux_value`` defaults to True because the fitted artifact cannot be consulted here; only
    ``AG_FETCH_PRETRAINED_WEIGHTS`` can tighten this.
    """
    if fetch_allowed(True, stage=stage):
        yield
        return

    import importlib

    modules = []
    for name in _PATCH_TARGETS:
        try:
            modules.append(importlib.import_module(name))
        except ImportError:  # pragma: no cover - layout differs across tabicl versions
            continue

    originals = [(m, m.hf_hub_download) for m in modules if hasattr(m, "hf_hub_download")]

    def _guarded(*args, **kwargs):
        # tabicl probes the cache with local_files_only=True first; that call touches no network.
        if kwargs.get("local_files_only"):
            for module, original in originals:
                if module.hf_hub_download is _guarded:
                    return original(*args, **kwargs)
        raise PretrainedWeightsUnavailableError(
            unavailable_message(model_name=model_name, stage=stage, location=kwargs.get("filename"))
        )

    for module, _ in originals:
        module.hf_hub_download = _guarded
    try:
        yield
    finally:
        for module, original in originals:
            module.hf_hub_download = original
