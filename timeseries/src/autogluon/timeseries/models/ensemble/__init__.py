from .abstract import AbstractTimeSeriesEnsembleModel
from .array_based import MedianEnsemble, PerQuantileTabularEnsemble, TabularEnsemble
from .weighted import GreedyEnsemble, PerformanceWeightedEnsemble, PerItemGreedyEnsemble, SimpleAverageEnsemble


def get_ensemble_class(name: str):
    mapping = {
        "GreedyEnsemble": GreedyEnsemble,
        "PerItemGreedyEnsemble": PerItemGreedyEnsemble,
        "PerformanceWeightedEnsemble": PerformanceWeightedEnsemble,
        "SimpleAverageEnsemble": SimpleAverageEnsemble,
        "WeightedEnsemble": GreedyEnsemble,  # old alias for this model
        "MedianEnsemble": MedianEnsemble,
        "TabularEnsemble": TabularEnsemble,
        "PerQuantileTabularEnsemble": PerQuantileTabularEnsemble,
    }
    if name not in mapping:
        raise ValueError(f"Unknown ensemble type: {name}. Available: {list(mapping.keys())}")
    return mapping[name]


__all__ = [
    "AbstractTimeSeriesEnsembleModel",
    "GreedyEnsemble",
    "PerItemGreedyEnsemble",
    "MedianEnsemble",
    "PerformanceWeightedEnsemble",
    "PerQuantileTabularEnsemble",
    "SimpleAverageEnsemble",
    "TabularEnsemble",
    "get_ensemble_class",
]
