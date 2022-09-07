from typing import Union, List, Any, Dict

from .. import AnalysisState
from ..analysis import AbstractAnalysis, DATASET_ARGS


class FeatureInteraction(AbstractAnalysis):

    def __init__(self,
                 x: Union[None, str] = None,
                 y: [None, str] = None,
                 hue: str = None,
                 parent: Union[None, AbstractAnalysis] = None,
                 children: List[AbstractAnalysis] = [],
                 **kwargs) -> None:
        super().__init__(parent, children, **kwargs)
        self.x = x
        self.y = y
        self.hue = hue

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs):
        cols = {
            'x': self.x,
            'y': self.y,
            'hue': self.hue,
        }
        cols = {k: v for k, v in cols.items() if v is not None}
        interactions: List[Dict[str, Any]] = state.get('interactions', [])

        for ds in DATASET_ARGS:
            if ds in args and args[ds] is not None:
                missing_cols = [c for c in cols.values() if c not in args[ds].columns]
                if len(missing_cols) == 0:
                    df = args[ds][cols.values()]
                    ds_interaction = {
                        'features': cols,
                        'dataset': ds,
                        'data': df,
                    }
                    interactions.append(ds_interaction)
        state.interactions = interactions
