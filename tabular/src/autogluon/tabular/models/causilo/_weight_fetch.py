"""Enforcement of ``ag.fetch_pretrained_weights`` for Causilo.

``causilo.checkpoints.load_pretrained_model`` resolves the pinned checkpoint with
``snapshot_download``, which reuses the Hugging Face cache but reaches the network on a miss. The
guard forces ``local_files_only=True`` on that call: a cached checkpoint still resolves, and only a
real fetch is refused.
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
    """No-op when fetching is allowed; otherwise restrict checkpoint resolution to the local cache."""
    if fetch_allowed(aux_value, stage=stage):
        yield
        return

    import causilo.checkpoints as checkpoints

    original = checkpoints.snapshot_download

    def _local_only(*args, **kwargs):
        from huggingface_hub.errors import LocalEntryNotFoundError

        kwargs["local_files_only"] = True
        try:
            return original(*args, **kwargs)
        except LocalEntryNotFoundError as err:
            location = args[0] if args else kwargs.get("repo_id")
            raise PretrainedWeightsUnavailableError(
                unavailable_message(model_name=model_name, stage=stage, location=location)
            ) from err

    checkpoints.snapshot_download = _local_only
    try:
        yield
    finally:
        checkpoints.snapshot_download = original
