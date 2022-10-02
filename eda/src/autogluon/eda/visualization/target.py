from typing import Union, Dict, Any

from autogluon.eda import AnalysisState
from autogluon.eda.visualization import AbstractVisualization, SimpleVerticalLinearLayout, SimpleHorizontalLayout
from ..visualization import FeatureInteractionVisualization, CorrelationVisualization, DatasetStatistics
from ..visualization.layouts import MarkdownSectionComponent


class TargetFacetAnalysisVisualization(AbstractVisualization):

    def __init__(self,
                 namespace: str = None,
                 fig_args: Union[None, Dict[str, Any]] = {},
                 viz_args: Union[None, Dict[str, Any]] = {},
                 **kwargs) -> None:
        super().__init__(namespace, **kwargs)
        self.fig_args = fig_args
        self.viz_args = viz_args

    def can_handle(self, state: AnalysisState) -> bool:
        return self.all_keys_must_be_present(state, ['target_correlations', 'target_statistics'])

    def _render(self, state: AnalysisState) -> None:
        viz_facets = []
        corr_vars = state.target_correlations.high_corr_vars

        if corr_vars is None:
            corr_vars = {}
            corr_facets = []
        else:
            corr_facets = [CorrelationVisualization(
                namespace='target_correlations', fig_args=self.fig_args.get('correlation', {}), headers=True, **self.viz_args.get('correlation', {})
            )]

        for var, corr in corr_vars.items():
            if var == state.target_correlations.label:
                continue

            viz_facets.append(MarkdownSectionComponent(f'### Variable with high correlation vs {state.target_correlations.label} - {var}: {corr:.2f}'))
            viz_facets.append(
                SimpleHorizontalLayout(facets=[
                    FeatureInteractionVisualization(
                        namespace=f'target_correlations.{var}',
                        render_only_idx=i,
                        fig_args=self.fig_args.get(f'target_correlations.{var}', {}),
                        chart_args=self.viz_args.get(f'target_correlations.{var}', {})
                    ) for i in [0, 1]]
                )
            )
            viz_facets.append(MarkdownSectionComponent('---'))

        SimpleVerticalLinearLayout(
            facets=[
                MarkdownSectionComponent(f'## Variable analysis: {state.target_correlations.label}'),
                FeatureInteractionVisualization(
                    namespace='target_interaction',
                    fig_args=self.fig_args.get('target_distribution', {}),
                    **self.viz_args.get('target_distribution', {})
                ),
                DatasetStatistics(namespace='target_statistics', headers=False, **self.viz_args.get('dataset_statistics', {})),
                *corr_facets,
                *viz_facets,
            ],
        ).render(state)

    class ParametersBuilder:
        __fig_args = {}
        __viz_args = {}

        def __assign_param_to_subkey(self, o: dict, key_1, key_2, param):
            if key_1 not in o:
                o[key_1] = {}
            o[key_1][key_2] = param

        def with_field_subplot_args(self, field: str, chart_no: int, **kwargs):
            self.__assign_param_to_subkey(self.__fig_args, f'target_correlations.{field}', chart_no, kwargs)
            return self

        def with_field_chart_args(self, field: str, chart_no: int, **kwargs):
            self.__assign_param_to_subkey(self.__viz_args, f'target_correlations.{field}', chart_no, kwargs)
            return self

        def with_correlation_subplot_args(self, **kwargs):
            self.__fig_args['correlation'] = kwargs
            return self

        def with_correlation_chart_args(self, **kwargs):
            self.__viz_args['correlation'] = kwargs
            return self

        def with_target_distribution_subplot_args(self, **kwargs):
            self.__assign_param_to_subkey(self.__fig_args, 'target_distribution', 0, kwargs)
            return self

        def with_target_distribution_chart_args(self, **kwargs):
            self.__assign_param_to_subkey(self.__viz_args, 'target_distribution', 0, kwargs)
            return self

        def build(self):
            return dict(fig_args=self.__fig_args, viz_args=self.__viz_args)
