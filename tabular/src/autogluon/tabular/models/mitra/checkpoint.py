"""Checkpoint identities are independent of the shared Tab2D architecture."""

from __future__ import annotations

import os

DEFAULT_MODEL_VERSION = "2"
CHECKPOINTS = {
    "1": {
        "classification": "autogluon/mitra-classifier",
        "regression": "autogluon/mitra-regressor",
    },
    "2": {
        "classification": "autogluon/mitra-classifier-2",
        "regression": "autogluon/mitra-regressor-2",
    },
}


def validate_model_version(model_version: str) -> str:
    if not isinstance(model_version, str) or model_version not in CHECKPOINTS:
        raise ValueError(f"Unknown Mitra model_version={model_version!r}. Expected '1' or '2'.")
    return model_version


def checkpoint_version(path_or_repo_id: str, metadata_version: str | None = None) -> str | None:
    """Read release identity from explicit metadata or a known Hub ID, never tensor shapes."""
    known_version = None
    if not os.path.isdir(path_or_repo_id):
        for version, checkpoints in CHECKPOINTS.items():
            if str(path_or_repo_id) in checkpoints.values():
                known_version = version
                break
        if str(path_or_repo_id) == "autogluon/mitra-classifier-1.1":
            known_version = "1"
    if metadata_version is not None:
        validate_model_version(metadata_version)
        if known_version is not None and metadata_version != known_version:
            raise ValueError(f"Checkpoint {path_or_repo_id!r} has conflicting Mitra version metadata.")
        return metadata_version
    return known_version


def validate_checkpoint_version(
    path_or_repo_id: str, model_version: str, metadata_version: str | None = None
) -> str | None:
    validate_model_version(model_version)
    version = checkpoint_version(path_or_repo_id, metadata_version)
    if version is not None and version != model_version:
        raise ValueError(
            f"Checkpoint {str(path_or_repo_id)!r} is Mitra v{version}, but model_version={model_version!r} "
            f"selects v{model_version} weights. Set model_version={version!r} "
            "or choose a checkpoint for the selected version."
        )
    return version
