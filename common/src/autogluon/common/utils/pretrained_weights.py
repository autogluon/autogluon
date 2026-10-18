"""Policy for fetching pretrained weights that are absent from the local cache.

Models carrying pretrained weights download them on demand. That is convenient on a workstation
and unwanted in places where a fit must not reach the network: an air-gapped or VPC-isolated host,
a benchmark run that must not vary with what a remote registry serves, or a serving container whose
image was built with the weights baked in and should fail loudly rather than silently re-download.

The policy is expressed by the ``ag.fetch_pretrained_weights`` auxiliary parameter and can be
overridden per-process by :data:`FETCH_ENV_VAR`. Enforcement is per-package, because each
foundation-model library reaches the network differently; this module only decides *whether* a
fetch is allowed.
"""

from __future__ import annotations

import os

__all__ = [
    "FETCH_ENV_VAR",
    "FIT_ONLY",
    "PretrainedWeightsUnavailableError",
    "fetch_allowed",
    "resolve_policy",
]

#: Deployment-level override, mirroring ``AG_ALLOW_PICKLE_FROM_URL``. A serving host sets this to
#: constrain fetches regardless of what a fitted artifact was pickled with.
FETCH_ENV_VAR = "AG_FETCH_PRETRAINED_WEIGHTS"

#: Fetch while fitting, never when loading a saved model.
FIT_ONLY = "fit_only"

_VALID = (True, False, FIT_ONLY)


class PretrainedWeightsUnavailableError(RuntimeError):
    """Weights are absent from the local cache and the active policy forbids fetching them."""


def resolve_policy(aux_value: bool | str) -> bool | str:
    """Resolve the effective policy, letting the environment override the fitted artifact.

    The environment wins because an inference-time policy belongs to the deployment rather than to
    the model: a serving host must be able to forbid fetches without re-fitting.
    """
    raw = os.environ.get(FETCH_ENV_VAR)
    if raw is not None:
        lowered = raw.strip().lower()
        if lowered in ("true", "1"):
            return True
        if lowered in ("false", "0"):
            return False
        if lowered == FIT_ONLY:
            return FIT_ONLY
        raise ValueError(f"{FETCH_ENV_VAR}={raw!r} is not understood; expected True, False, or {FIT_ONLY!r}.")
    if aux_value not in _VALID:
        raise ValueError(
            f"ag.fetch_pretrained_weights={aux_value!r} is not understood; expected True, False, or {FIT_ONLY!r}."
        )
    return aux_value


def fetch_allowed(aux_value: bool | str, *, stage: str) -> bool:
    """Whether a remote fetch is permitted at ``stage``, one of ``"fit"`` or ``"load"``."""
    if stage not in ("fit", "load"):
        raise ValueError(f"stage must be 'fit' or 'load', got {stage!r}")
    policy = resolve_policy(aux_value)
    if policy is True:
        return True
    if policy is False:
        return False
    return stage == "fit"


def unavailable_message(*, model_name: str, stage: str, location: object = None) -> str:
    """A consistent error message across packages."""
    where = f"\nExpected location: {location}" if location is not None else ""
    return (
        f"{model_name} needs pretrained weights that are not in the local cache, and fetching them "
        f"is disabled at {stage} time (ag.fetch_pretrained_weights / {FETCH_ENV_VAR}).{where}\n"
        f"Either pre-populate the cache on this machine, or allow fetching."
    )
