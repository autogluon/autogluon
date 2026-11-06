"""Default Hugging Face checkpoint choices for the native Mitra API."""

from __future__ import annotations

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
