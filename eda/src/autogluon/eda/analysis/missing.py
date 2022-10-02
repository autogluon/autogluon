from typing import Union, List

import pandas as pd

from . import AnalysisState, StateCheckMixin, MissingValuesAnalysis
from ..analysis import AbstractAnalysis, BaseAnalysis, Correlation, TargetFacetAnalysis
from ..analysis import Namespace


class MissingValuesFacetAnalysis(AbstractAnalysis, StateCheckMixin):

    def __init__(self,
                 use_all_datasets=False,
                 correlation_method='spearman',
                 correlations_focus_threshold=0.5,
                 parent: Union[None, AbstractAnalysis] = None,
                 children: List[AbstractAnalysis] = [],
                 **kwargs) -> None:
        assert len(children) == 0, f'{self.__name__} cannot have child analyses'
        super().__init__(parent, children, **kwargs)
        self.correlations_focus_threshold = correlations_focus_threshold
        self.correlation_method = correlation_method
        self.use_all_datasets = use_all_datasets

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return self.all_keys_must_be_present(args, ['train_data', 'label'])

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        if self.use_all_datasets:
            df = pd.concat([df.drop(columns=args.label, errors='ignore') for _, df in self.available_datasets(args)], axis=0, ignore_index=True)
        else:
            df = args.train_data.drop(columns=args.label, errors='ignore')

        state = self._initial_missing_values_analysis(df, args, state, fit_kwargs)
        for section in ['mid_missing_counts', 'low_missing_counts']:
            state = self._identify_highly_correlated_fields_vs_missing_field(df, section, fit_kwargs, state)
            for field in state.missing_fields_correlations[section].keys():
                a = BaseAnalysis(children=[
                    Namespace(namespace=f'missing_values_field_analysis', train_data=df, label=field, children=[
                        Namespace(namespace=section, children=[
                            Namespace(namespace=field, children=[
                                TargetFacetAnalysis(
                                    correlation_method=self.correlation_method,
                                    correlations_focus_threshold=self.correlations_focus_threshold
                                )
                            ])
                        ])
                    ])
                ])
                a.state = state
                state = a.fit(**fit_kwargs)

    def _initial_missing_values_analysis(self, df, args, state, fit_kwargs):
        a = BaseAnalysis(children=[
            Namespace(namespace='missing_values', train_data=df, label=args.label, children=[
                MissingValuesAnalysis(),
            ]),
        ])
        a.state = state
        state = a.fit(**fit_kwargs)
        return state

    def _identify_highly_correlated_fields_vs_missing_field(self, df, section, fit_kwargs, state):
        allowed_keys = ['low_missing_counts', 'mid_missing_counts', 'high_missing_counts']
        assert section in allowed_keys, f'section must be in {allowed_keys}'
        corr_analyses = []
        corr_args = {**dict(method=self.correlation_method, focus_field_threshold=self.correlations_focus_threshold)}
        for field in state.missing_values.missing_statistics.train_data[section]:
            facet = Namespace(namespace=field, train_data=df[df[field].notna()], children=[
                Correlation(focus_field=field, **corr_args)
            ])
            corr_analyses.append(facet)
        a = BaseAnalysis(children=[
            Namespace(namespace='missing_fields_correlations', children=[
                Namespace(namespace=section, children=[
                    *corr_analyses,
                ]),
            ]),
        ])
        a.state = state
        state = a.fit(**fit_kwargs)
        return state
