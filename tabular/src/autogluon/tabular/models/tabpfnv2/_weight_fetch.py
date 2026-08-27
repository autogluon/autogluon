"""Policy gate for fetching TabPFN's pretrained weights from a remote source.

``tabpfn`` already has the right primitive -- ``load_model_criterion_config(...,
download_if_not_exists=...)`` -- but ``tabpfn.base`` hardcodes it to ``True`` (see
``tabpfn/base.py``), so the estimator API offers no way to turn fetching off. Until that is
exposed upstream, gate the single function that performs the fetch.

``HF_HUB_OFFLINE`` is not sufficient: ``download_model`` falls back to
``urllib.request.urlopen`` against huggingface.co when the ``huggingface_hub`` path fails, so an
HF-level switch only changes *which* code path reaches the network.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Iterator

logger = logging.getLogger(__name__)

#: Deployment-level override, mirroring ``AG_ALLOW_PICKLE_FROM_URL``. A serving host sets this to
#: forbid fetches regardless of what the fitted artifact was pickled with.
FETCH_ENV_VAR = "AG_FETCH_PRETRAINED_WEIGHTS"

FIT_ONLY = "fit_only"


class PretrainedWeightsUnavailableError(RuntimeError):
    """Weights are absent locally and the active policy forbids fetching them."""


def resolve_policy(aux_value: bool | str) -> bool | str:
    """Environment beats the fitted artifact, matching the ``AG_ALLOW_PICKLE_FROM_URL`` precedent."""
    raw = os.environ.get(FETCH_ENV_VAR)
    if raw is None:
        return aux_value
    lowered = raw.strip().lower()
    if lowered in ("true", "1"):
        return True
    if lowered in ("false", "0"):
        return False
    if lowered == FIT_ONLY:
        return FIT_ONLY
    raise ValueError(f"{FETCH_ENV_VAR}={raw!r} is not understood; expected one of True, False, {FIT_ONLY!r}.")


def fetch_allowed(aux_value: bool | str, *, stage: str) -> bool:
    """Whether a remote fetch is permitted at ``stage`` ("fit" or "load")."""
    policy = resolve_policy(aux_value)
    if policy is True:
        return True
    if policy is False:
        return False
    if policy == FIT_ONLY:
        return stage == "fit"
    raise ValueError(
        f"ag.fetch_pretrained_weights={policy!r} is not understood; expected True, False, or {FIT_ONLY!r}."
    )


@contextlib.contextmanager
def no_remote_fetch(*, stage: str, model_name: str) -> Iterator[None]:
    """Make a missing checkpoint raise instead of silently reaching the network."""
    import tabpfn.model_loading as model_loading

    original = model_loading.download_model

    def _blocked(to, *args, **kwargs):
        raise PretrainedWeightsUnavailableError(
            f"{model_name} needs pretrained weights that are not in the local cache, and "
            f"fetching them is disabled at {stage} time "
            f"(ag.fetch_pretrained_weights / {FETCH_ENV_VAR}).\n"
            f"Expected location: {to}\n"
            f"Either pre-populate the cache on this machine, or allow fetching."
        )

    model_loading.download_model = _blocked
    try:
        yield
    finally:
        model_loading.download_model = original


@contextlib.contextmanager
def weight_fetch_policy(aux_value: bool | str, *, stage: str, model_name: str) -> Iterator[None]:
    """No-op when fetching is allowed; otherwise block it for the duration."""
    if fetch_allowed(aux_value, stage=stage):
        yield
    else:
        with no_remote_fetch(stage=stage, model_name=model_name):
            yield
