import warnings

gluonts_json_warning = (
    "Using `json`-module for json-handling. "
    "Consider installing one of `orjson`, `ujson` "
    "to speed up serialization and deserialization."
)
warnings.filterwarnings("ignore", message=gluonts_json_warning)

from .torch import DeepARModel, SimpleFeedForwardModel

__all__ = ["DeepARModel", "SimpleFeedForwardModel"]
