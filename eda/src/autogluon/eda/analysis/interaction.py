from typing import Union, List, Any, Dict

from .base import AbstractAnalysis
from .. import AnalysisState

__all__ = ['Correlation', 'CorrelationSignificance']


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

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return True

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

    def __init__(self, method='spearman',
                 focus_field: Union[None, str] = None,
                 focus_field_threshold: float = 0.5,
                 parent: Union[None, AbstractAnalysis] = None,
                 children: List[AbstractAnalysis] = [],
                 **kwargs) -> None:
        self.method = method
        self.focus_field = focus_field
        self.focus_field_threshold = focus_field_threshold
        super().__init__(parent, children, **kwargs)

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return True

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs):
        state.correlations = {}
        state.correlations_method = self.method
        for (ds, df) in self.available_datasets(args):
            if self.method == 'phik':
                state.correlations[ds] = df.phik_matrix(**self.args, verbose=False)
            else:
                args = {}
                if self.method is not None:
                    args['method'] = self.method
                state.correlations[ds] = df.corr(**args, **self.args)

            if self.focus_field is not None and self.focus_field in state.correlations[ds].columns:
                state.correlations_focus_field = self.focus_field
                state.correlations_focus_field_threshold = self.focus_field_threshold
                state.correlations_focus_high_corr = {}
                df_corr = state.correlations[ds]
                df_corr = df_corr[df_corr[self.focus_field] >= self.focus_field_threshold]
                keep_cols = df_corr.index.values
                state.correlations[ds] = df_corr[keep_cols]

                high_corr = state.correlations[ds][[self.focus_field]].sort_values(self.focus_field, ascending=False).drop(self.focus_field)
                state.correlations_focus_high_corr[ds] = high_corr


class CorrelationSignificance(AbstractAnalysis):

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return 'correlations' in state and 'correlations_method' in state

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs):
        state.significance_matrix = {}
        for (ds, df) in self.available_datasets(args):
            state.significance_matrix[ds] = df[state.correlations[ds].columns].significance_matrix(**self.args, verbose=False)
