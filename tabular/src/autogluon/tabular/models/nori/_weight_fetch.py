"""Enforcement of ``ag.fetch_pretrained_weights`` for Nori at inference time.

``NoriRegressor`` does not hold the pretrained module: it builds its predictor lazily on the first
``predict``, resolving the checkpoint then. So unlike the other foundation models, gating ``fit``
alone leaves the inference-time fetch open -- a model loaded on a cold serving host would download
weights on its first prediction.

``synthefy_nori.hf.download_checkpoint`` is the single resolution point, and the library imports it
inside the calling function, so replacing the module attribute is enough.
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
    """Restrict Nori's checkpoint resolution to the local cache."""
    import synthefy_nori.hf as nori_hf

    original = nori_hf.download_checkpoint

    def _cached_only(repo_id=None, filename=None, *, model=None, **kwargs):
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        repo = nori_hf.resolve_model_repo(model) if model is not None else (repo_id or nori_hf.DEFAULT_MODEL_REPO_ID)
        name = filename or nori_hf.DEFAULT_CHECKPOINT_FILENAME
        try:
            return hf_hub_download(repo_id=repo, filename=name, local_files_only=True)
        except LocalEntryNotFoundError as err:
            raise PretrainedWeightsUnavailableError(
                unavailable_message(model_name=model_name, stage=stage, location=f"{repo}/{name}")
            ) from err

    nori_hf.download_checkpoint = _cached_only
    try:
        yield
    finally:
        nori_hf.download_checkpoint = original
