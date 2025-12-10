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

    for full_name, ensemble_class in list(mapping.items()):
        if full_name.endswith("Ensemble"):
            alias = full_name[:-8]  # Remove "Ensemble" suffix
            mapping[alias] = ensemble_class

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
