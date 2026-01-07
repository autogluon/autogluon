from .abstract import AbstractTimeSeriesEnsembleModel
from .array_based import LinearStackerEnsemble, MedianEnsemble, PerQuantileTabularEnsemble, TabularEnsemble
from .per_item_greedy import PerItemGreedyEnsemble
from .weighted import GreedyEnsemble, PerformanceWeightedEnsemble, SimpleAverageEnsemble


def get_ensemble_class(name: str):
    mapping = {
        "Greedy": GreedyEnsemble,
        "PerItemGreedy": PerItemGreedyEnsemble,
        "PerformanceWeighted": PerformanceWeightedEnsemble,
        "SimpleAverage": SimpleAverageEnsemble,
        "Weighted": GreedyEnsemble,  # old alias for this model
        "Median": MedianEnsemble,
        "Tabular": TabularEnsemble,
        "PerQuantileTabular": PerQuantileTabularEnsemble,
        "LinearStacker": LinearStackerEnsemble,
    }

    name_clean = name.removesuffix("Ensemble")
    if name_clean not in mapping:
        raise ValueError(f"Unknown ensemble type: {name}. Available: {list(mapping.keys())}")
    return mapping[name_clean]


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
