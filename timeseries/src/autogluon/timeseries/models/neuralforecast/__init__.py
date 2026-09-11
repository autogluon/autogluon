"""NeuralForecast model aliases; importing this module does not import the backend."""

from ._catalog import NEURALFORECAST_MODELS, NEURALFORECAST_REPOSITORY, NEURALFORECAST_REVISION
from .model import NeuralForecastModel

__all__ = ["NeuralForecastModel", "NEURALFORECAST_MODELS", "NEURALFORECAST_REPOSITORY", "NEURALFORECAST_REVISION"]

# All classes live in this importable module, which preserves pickle/load identity.
for _name in NEURALFORECAST_MODELS:
    _class_name = f"NF{_name}Model"
    globals()[_class_name] = type(
        _class_name,
        (NeuralForecastModel,),
        {"__module__": __name__, "nf_model_name": _name, "__doc__": f"Pinned NeuralForecast {_name} adapter."},
    )
    __all__.append(_class_name)

del _name, _class_name
