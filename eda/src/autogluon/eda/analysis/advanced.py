from typing import Union, List

import pandas as pd

from .base import AbstractAnalysis, BaseAnalysis, Namespace
from .base import AnalysisState, StateCheckMixin
from .dataset import RawTypesAnalysis, DatasetSummary, VariableTypeAnalysis, DistributionFit
from .interaction import FeatureInteraction, Correlation
from .missing import MissingValuesAnalysis


class TargetFacetAnalysis(AbstractAnalysis, StateCheckMixin):

    def __init__(self,
                 correlation_method='spearman',
                 correlations_focus_threshold=0.75,
                 parent: Union[None, AbstractAnalysis] = None,
                 children: List[AbstractAnalysis] = [],
                 fit_distributions: bool = False,
                 **kwargs) -> None:
        assert len(children) == 0, f'{self.__name__} cannot have child analyses'
        super().__init__(parent, children, **kwargs)
        self.correlations_focus_threshold = correlations_focus_threshold
        self.correlation_method = correlation_method
        self.fit_distributions = fit_distributions

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return self.all_keys_must_be_present(args, 'train_data', 'label')

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        corr_args = {**dict(method=self.correlation_method, focus_field=args.label, focus_field_threshold=self.correlations_focus_threshold),
                     **self.args.get('correlation', {})}

        target_statistics_facets = [
            RawTypesAnalysis(),
            VariableTypeAnalysis(),
            DatasetSummary(),
            MissingValuesAnalysis(),
        ]
        if self.fit_distributions:
            target_statistics_facets.append(
                DistributionFit(columns=args.label)
            )
        target_interation_facets = [
            RawTypesAnalysis(),
            FeatureInteraction(x=args.label),
        ]

        a = BaseAnalysis(state=state, children=[
            Namespace(namespace='target_correlations', train_data=args.train_data, children=[
                RawTypesAnalysis(),
                VariableTypeAnalysis(),
            ]),
        ])
        state = a.fit(**fit_kwargs)

        type = state.target_correlations.variable_type.train_data[args.label]
        df = args.train_data
        if type == 'category':
            data = df.copy()
            data[args.label] = data[args.label].fillna('--NA--')
        else:
            data = df

        a = BaseAnalysis(state=state, children=[
            Namespace(namespace='target_interaction', train_data=data[[args.label]], children=target_interation_facets),
            Namespace(namespace='target_statistics', train_data=args.train_data[[args.label]], children=target_statistics_facets),
            Namespace(namespace='target_correlations', train_data=args.train_data, children=[
                Correlation(**corr_args),
            ]),
        ])
        state = a.fit(**fit_kwargs)

        state.target_correlations.label = args.label

        high_corr_data = state.target_correlations.correlations_focus_high_corr
        if high_corr_data is None:
            return
        state.target_correlations.high_corr_vars = {var: corr for var, corr in high_corr_data.train_data[args.label].items()}

        anlz_facets = []
        for var, corr in state.target_correlations.high_corr_vars.items():
            if var == args.label:
                continue
            anlz_facets.append(
                Namespace(namespace='target_correlations', children=[
                    Namespace(namespace=var, train_data=data, children=[
                        RawTypesAnalysis(),
                        FeatureInteraction(x=var),
                        FeatureInteraction(x=var, y=args.label)
                    ])
                ])
            )

        a = BaseAnalysis(state=state, children=anlz_facets)
        a.fit(**fit_kwargs)


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
        return self.all_keys_must_be_present(args, 'train_data', 'label')

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        if self.use_all_datasets:
            df = pd.concat([df.drop(columns=args.label, errors='ignore') for _, df in self.available_datasets(args)], axis=0, ignore_index=True)
        else:
            df = args.train_data.drop(columns=args.label, errors='ignore')

        state = self._initial_missing_values_analysis(df, args, state, fit_kwargs)
        for section in ['low_missing_counts', 'mid_missing_counts', 'high_missing_counts']:
            state = self._identify_highly_correlated_fields_vs_missing_field(df, section, fit_kwargs, state)
            for field in state.missing_fields_correlations[section].keys():
                a = BaseAnalysis(state=state, children=[
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
                state = a.fit(**fit_kwargs)

    def _initial_missing_values_analysis(self, df, args, state, fit_kwargs):
        a = BaseAnalysis(state=state, children=[
            Namespace(namespace='missing_values', train_data=df, label=args.label, children=[
                MissingValuesAnalysis(),
            ]),
        ])
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
        a = BaseAnalysis(state=state, children=[
            Namespace(namespace='missing_fields_correlations', children=[
                Namespace(namespace=section, children=[
                    *corr_analyses,
                ]),
            ]),
        ])
        state = a.fit(**fit_kwargs)
        return state
