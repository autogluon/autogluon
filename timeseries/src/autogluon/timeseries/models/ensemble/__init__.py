from .abstract import AbstractTimeSeriesEnsembleModel
from .array_based import LinearStackerEnsemble, MedianEnsemble, PerQuantileTabularEnsemble, TabularEnsemble
from .per_item_greedy import PerItemGreedyEnsemble
from .weighted import GreedyEnsemble, PerformanceWeightedEnsemble, SimpleAverageEnsemble


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
        "LinearStackerEnsemble": LinearStackerEnsemble,
    }
    if name not in mapping:
        raise ValueError(f"Unknown ensemble type: {name}. Available: {list(mapping.keys())}")
    return mapping[name]


__all__ = [
    "AbstractTimeSeriesEnsembleModel",
    "GreedyEnsemble",
    "LinearStackerEnsemble",
    "MedianEnsemble",
    "PerformanceWeightedEnsemble",
    "PerItemGreedyEnsemble",
    "PerQuantileTabularEnsemble",
    "SimpleAverageEnsemble",
    "TabularEnsemble",
    "get_ensemble_class",
]
