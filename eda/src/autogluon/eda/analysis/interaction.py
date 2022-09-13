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


class Correlation(AbstractAnalysis):

    def __init__(self, method='spearman', significance=False, parent: Union[None, AbstractAnalysis] = None, children: List[AbstractAnalysis] = [], **kwargs) -> None:
        self.method = method
        self.significance = significance
        super().__init__(parent, children, **kwargs)

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs):
        state.correlations = {}
        state.significance_matrix = {}
        state.correlations_method = self.method
        if self.significance:
            state.significance_matrix = {}
        for (ds, df) in self.available_datasets(args):
            if self.method == 'phik':
                import phik  # required
                state.correlations[ds] = df.phik_matrix(**self.args, verbose=False)
                if self.significance:
                    state.significance_matrix[ds] = df.significance_matrix(**self.args, verbose=False)

            else:
                args = {}
                if self.method is not None:
                    args['method'] = self.method
                state.correlations[ds] = df.corr(**args, **self.args)
                if self.significance:
                    state.significance_matrix[ds] = df[state.correlations[ds].columns].significance_matrix(**self.args, verbose=False)
