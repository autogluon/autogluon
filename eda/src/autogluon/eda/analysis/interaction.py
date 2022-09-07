from typing import Union, List, Any, Dict

from .. import AnalysisState
from ..analysis import AbstractAnalysis


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

        for (ds, df) in self.available_datasets(args):
            missing_cols = [c for c in cols.values() if c not in df.columns]
            if len(missing_cols) == 0:
                df = df[cols.values()]
                ds_interaction = {
                    'features': cols,
                    'dataset': ds,
                    'data': df,
                }
                interactions.append(ds_interaction)
        state.interactions = interactions

#
# class Correlation(AbstractAnalysis):
#     def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs):
#         state.correlations = {}
#         with self.available_datasets(args) as (ds, df):
#             state.correlations[ds] = args[ds]
