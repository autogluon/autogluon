from typing import Union, List

from . import AnalysisState, StateCheckMixin
from ..analysis.dataset import MissingValuesAnalysis
from ..analysis import AbstractAnalysis, BaseAnalysis
from ..analysis import RawTypesAnalysis, FeatureInteraction, Correlation, Namespace, DatasetSummary, VariableTypeAnalysis, \
    DistributionFit


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
        return self.all_keys_must_be_present(args, ['train_data', 'label'])

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

        a = BaseAnalysis(children=[
            Namespace(namespace='target_correlations', train_data=args.train_data, children=[
                RawTypesAnalysis(),
                VariableTypeAnalysis(),
            ]),
        ])
        a.state = state
        state = a.fit(**fit_kwargs)

        type = state.target_correlations.variable_type.train_data[args.label]
        df = args.train_data
        if type == 'category':
            data = df.copy()
            data[args.label] = data[args.label].fillna('--NA--')
        else:
            data = df

        a = BaseAnalysis(children=[
            Namespace(namespace='target_interaction', train_data=data[[args.label]], children=target_interation_facets),
            Namespace(namespace='target_statistics', train_data=args.train_data[[args.label]], children=target_statistics_facets),
            Namespace(namespace='target_correlations', train_data=args.train_data, children=[
                Correlation(**corr_args),
            ]),
        ])
        a.state = state
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

        a = BaseAnalysis(children=anlz_facets)
        a.state = state
        a.fit(**fit_kwargs)
