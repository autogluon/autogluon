"""Enforcement of ``ag.fetch_pretrained_weights`` for TabFM.

``tabfm``'s PyTorch loader resolves its checkpoint through ``huggingface_hub``'s model mixin,
which downloads a config file and then a snapshot of the checkpoint folder. Rather than intercept
each call, the guard puts ``huggingface_hub`` in offline mode for the duration: every request then
resolves from the local cache, so a cached checkpoint still loads and only a real fetch is refused.
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
    """No-op when fetching is allowed; otherwise resolve TabFM's checkpoint from the local cache only."""
    if fetch_allowed(aux_value, stage=stage):
        yield
        return

    from huggingface_hub import constants
    from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled

    original = constants.HF_HUB_OFFLINE
    constants.HF_HUB_OFFLINE = True
    try:
        yield
    except (LocalEntryNotFoundError, OfflineModeIsEnabled) as err:
        from tabfm.src.pytorch.tabfm_v1_0_0 import HF_REPO_ID

        raise PretrainedWeightsUnavailableError(
            unavailable_message(model_name=model_name, stage=stage, location=HF_REPO_ID)
        ) from err
    finally:
        constants.HF_HUB_OFFLINE = original
