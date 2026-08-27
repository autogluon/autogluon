"""Enforcement of ``ag.fetch_pretrained_weights`` for TabDPT.

:class:`TabDPTModel` pins a checkpoint only when ``_checkpoint_filename`` is set; otherwise it
lets the library resolve its own default through ``TabDPTEstimator.download_weights``, which calls
``hf_hub_download`` with no cache probe. Blanket-blocking that call would break an already
provisioned machine, so the guard instead forces ``local_files_only=True``: a cached checkpoint
still resolves, and only a real fetch is refused.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

from autogluon.common.utils.pretrained_weights import (
    PretrainedWeightsUnavailableError,
    unavailable_message,
)

__all__ = ["local_weights_only"]


@contextlib.contextmanager
def local_weights_only(*, stage: str, model_name: str) -> Iterator[None]:
    """Restrict ``tabdpt``'s checkpoint resolution to the local cache."""
    import tabdpt.estimator as estimator

    original = estimator.hf_hub_download

    def _local_only(*args, **kwargs):
        from huggingface_hub.errors import LocalEntryNotFoundError

        kwargs["local_files_only"] = True
        try:
            return original(*args, **kwargs)
        except LocalEntryNotFoundError as err:
            raise PretrainedWeightsUnavailableError(
                unavailable_message(model_name=model_name, stage=stage, location=kwargs.get("filename"))
            ) from err

    estimator.hf_hub_download = _local_only
    try:
        yield
    finally:
        estimator.hf_hub_download = original
