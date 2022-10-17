from typing import Union, List, Any, Dict

import numpy as np
from scipy.cluster import hierarchy as hc
from scipy.stats import spearmanr

from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from .base import AbstractAnalysis
from .. import AnalysisState

__all__ = ['Correlation', 'CorrelationSignificance']

from ..state import StateCheckMixin


class FeatureInteraction(AbstractAnalysis):

    def __init__(self,
                 x: Union[None, str] = None,
                 y: [None, str] = None,
                 hue: str = None,
                 key: str = None,
                 parent: Union[None, AbstractAnalysis] = None,
                 children: List[AbstractAnalysis] = [],
                 **kwargs) -> None:
        super().__init__(parent, children, **kwargs)
        self.x = x
        self.y = y
        self.hue = hue
        self.key = key

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return self.all_keys_must_be_present(state, 'raw_type')

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs):
        cols = {
            'x': self.x,
            'y': self.y,
            'hue': self.hue,
        }

        if self.key is None:
            # if key is not provided, then convert to form: 'x:A|y:B|hue:C'; if values is not provided, then skip the value
            self.key = '|'.join([f'{k}:{v}' for k, v in {k: cols[k] for k in cols.keys()}.items() if v is not None])
        cols = {k: v for k, v in cols.items() if v is not None}

        interactions: Dict[str, Dict[str, Any]] = state.get('interactions', {})
        for (ds, df) in self.available_datasets(args):
            missing_cols = [c for c in cols.values() if c not in df.columns]
            if len(missing_cols) == 0:
                df = df[cols.values()]
                interaction = {
                    'features': cols,
                    'data': df,
                }
                if ds not in interactions:
                    interactions[ds] = {}
                interactions[ds][self.key] = interaction
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
        return self.all_keys_must_be_present(state, 'correlations', 'correlations_method')

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs):
        state.significance_matrix = {}
        for (ds, df) in self.available_datasets(args):
            state.significance_matrix[ds] = df[state.correlations[ds].columns].significance_matrix(**self.args, verbose=False)


class FeatureDistanceAnalysis(AbstractAnalysis, StateCheckMixin):
    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return self.all_keys_must_be_present(args, 'train_data', 'label')

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        x = args.train_data.drop(labels=[args.label], axis=1)
        feature_generator = AutoMLPipelineFeatureGenerator(
            enable_numeric_features=True,
            enable_categorical_features=True,
            enable_datetime_features=False,
            enable_text_special_features=False,
            enable_text_ngram_features=False,
            enable_raw_text_features=False,
            enable_vision_features=False,
        )
        x_transformed = feature_generator.fit_transform(X=x)
        corr = np.round(spearmanr(x_transformed).correlation, 4)
        np.fill_diagonal(corr, 1)
        corr_condensed = hc.distance.squareform(1 - np.nan_to_num(corr))
        z = hc.linkage(corr_condensed, method='average')
        s = {
            'columns': x_transformed.columns,
            'linkage': z,
        }
        state['feature_distance'] = s
