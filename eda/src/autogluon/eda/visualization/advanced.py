from typing import Any, Dict, Union

from .base import AbstractVisualization
from .dataset import DatasetStatistics
from .interaction import FeatureInteractionVisualization, CorrelationVisualization
from .layouts import SimpleHorizontalLayout, MarkdownSectionComponent
from .layouts import SimpleVerticalLinearLayout, TabLayout
from .missing import MissingValues
from ..state import AnalysisState


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
        return self.all_keys_must_be_present(state, 'target_correlations', 'target_statistics')

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

        def __init__(self) -> None:
            self.__fig_args = {}
            self.__viz_args = {}

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


class MissingValuesFacetAnalysisVisualization(AbstractVisualization):
    def __init__(self,
                 namespace: str = None,
                 field_args: Union[None, Dict[str, Any]] = {},
                 fig_args: Union[None, Dict[str, Any]] = {},
                 viz_args: Union[None, Dict[str, Any]] = {},
                 **kwargs) -> None:
        super().__init__(namespace, **kwargs)
        self.field_args = field_args
        self.fig_args = fig_args
        self.viz_args = viz_args

    def can_handle(self, state: AnalysisState) -> bool:
        return True

    def _render(self, state: AnalysisState) -> None:
        summary_tabs = TabLayout(namespace='missing_values', facets={
            'summary': DatasetStatistics(sort_by='missing_ratio', sort_asc=False),

            'matrix': SimpleVerticalLinearLayout(facets=[
                MarkdownSectionComponent('Nullity matrix is a data-dense display which lets you quickly visually pick out patterns in data completion'),
                MissingValues(graph_type='matrix'),
                MarkdownSectionComponent(
                    'The sparkline at right summarizes the general shape of the data completeness and points out the rows with'
                    ' the maximum and minimum nullity in the dataset.'),
            ]),

            'bar': SimpleVerticalLinearLayout(facets=[
                MarkdownSectionComponent('Simple visualization of nullity by column'),
                MissingValues(graph_type='bar'),
            ]),

            'heatmap': SimpleVerticalLinearLayout(facets=[
                MarkdownSectionComponent(
                    'Correlation heatmap measures nullity correlation: how strongly the presence or absence of one variable affects the presence of another'),
                MissingValues(graph_type='heatmap'),
                MarkdownSectionComponent(
                    'Nullity correlation ranges from `-1` (if one variable appears the other definitely does not) to `0` (variables appearing or not appearing'
                    ' have no effect on one another) to `1` (if one variable appears the other definitely also does).'
                    '\n\n'
                    'Variables that are always full or always empty have no meaningful correlation, and so are silently removed from the visualization.'
                    '\n\n'
                    'Entries marked `<1` or `>-1` have a correlation that is close to being exactingly negative or positive, but is still not quite'
                    ' perfectly so. This points to a small number of records in the dataset which are erroneous. These cases will require special attention.'),
            ]),

            'dendrogram': SimpleVerticalLinearLayout(facets=[
                MarkdownSectionComponent(
                    'The dendrogram reveals more complex relationships vs pairwise analysis in the correlation heatmap'),
                MissingValues(graph_type='dendrogram'),
                MarkdownSectionComponent(
                    'Cluster leaves which are linked together at a distance of zero fully predict one another\'s presence: one variable might always be empty'
                    ' when another is filled, or they might always both be filled or both empty. The further the distance away, less correlation between the'
                    ' variables.'),
            ]),
        })

        SimpleVerticalLinearLayout(
            facets=[
                MarkdownSectionComponent('## Dataset Missing Values Summary'),
                summary_tabs,
            ],
        ).render(state)

        sections = {
            'low_missing_counts': 'Low',
            'mid_missing_counts': 'Medium',
            'high_missing_counts': 'High',
        }
        for section, title in sections.items():
            field_analysis_tabs = {}
            if section not in state.missing_values_field_analysis:
                continue
            for field in state.missing_values_field_analysis[section].keys():
                args = self.field_args.get(field, dict(fig_args={}, viz_args={}))
                fig_args = args['fig_args']
                viz_args = args['viz_args']
                field_analysis_tabs[field] = TargetFacetAnalysisVisualization(
                    namespace=f'missing_values_field_analysis.{section}.{field}', fig_args=fig_args, viz_args=viz_args
                )
            field_analysis_tabs = TabLayout(facets=field_analysis_tabs)

            SimpleVerticalLinearLayout(
                facets=[
                    MarkdownSectionComponent(f'## Missing Values Fields Details - {title} number of missing values'),
                    field_analysis_tabs,
                ],
            ).render(state)

    class ParametersBuilder:

        def __init__(self) -> None:
            self.__field_args = {}

        def with_field_details(self, field: str, parameters: TargetFacetAnalysisVisualization.ParametersBuilder):
            self.__field_args[field] = parameters.build()
            return self

        def build(self):
            return dict(field_args=self.__field_args)
