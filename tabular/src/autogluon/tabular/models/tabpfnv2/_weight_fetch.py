"""Enforcement of ``ag.fetch_pretrained_weights`` for TabPFN.

``tabpfn`` already has the right primitive -- ``load_model_criterion_config(...,
download_if_not_exists=...)`` -- but ``tabpfn/base.py`` hardcodes it to ``True``, so the estimator
API offers no way to turn fetching off. Until that is exposed upstream, gate the single function
that performs the fetch.

``HF_HUB_OFFLINE`` is not sufficient here: ``download_model`` falls back to
``urllib.request.urlopen`` against huggingface.co when the ``huggingface_hub`` path fails, so an
HF-level switch only changes *which* code path reaches the network.
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


@contextlib.contextmanager
def weight_fetch_policy(aux_value: bool | str, *, stage: str, model_name: str) -> Iterator[None]:
    """No-op when fetching is allowed; otherwise make a missing checkpoint raise."""
    if fetch_allowed(aux_value, stage=stage):
        yield
        return

    import tabpfn.model_loading as model_loading

    original = model_loading.download_model

    def _blocked(to, *args, **kwargs):
        raise PretrainedWeightsUnavailableError(unavailable_message(model_name=model_name, stage=stage, location=to))

    model_loading.download_model = _blocked
    try:
        yield
    finally:
        model_loading.download_model = original
