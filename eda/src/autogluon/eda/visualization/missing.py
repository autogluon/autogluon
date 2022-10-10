from typing import Any, Dict, Union

import matplotlib.pyplot as plt
import missingno as msno

from . import DatasetStatistics, MarkdownSectionComponent, TargetFacetAnalysisVisualization
from .base import AbstractVisualization
from .jupyter import JupyterMixin
from .layouts import SimpleVerticalLinearLayout, TabLayout
from .. import AnalysisState


class MissingValues(AbstractVisualization, JupyterMixin):
    __OPERATIONS_MAPPING = {
        'matrix': msno.matrix,
        'bar': msno.bar,
        'heatmap': msno.heatmap,
        'dendrogram': msno.dendrogram,
    }

    def __init__(self,
                 graph_type: str,
                 headers: bool = False,
                 fig_args: Union[None, Dict[str, Any]] = {},
                 namespace: str = None,
                 **kwargs) -> None:
        super().__init__(namespace, **kwargs)
        self.graph_type = graph_type
        assert self.graph_type in self.__OPERATIONS_MAPPING, f'{self.graph_type} must be one of {self.__OPERATIONS_MAPPING.keys()}'
        self.headers = headers
        self.fig_args = fig_args

    def can_handle(self, state: AnalysisState) -> bool:
        return 'missing_statistics' in state

    def _render(self, state: AnalysisState) -> None:
        for ds, data in state.missing_statistics.items():
            self.render_header_if_needed(state, f'{ds} missing values analysis')
            widget = self.__OPERATIONS_MAPPING[self.graph_type]
            ax = widget(data.data, **self._kwargs)
            plt.show(ax)


class MissingValuesFieldVisualization(AbstractVisualization):

    def __init__(self,
                 field: str,
                 namespace: str = None,
                 **kwargs) -> None:
        super().__init__(namespace, **kwargs)
        self.field = field

    def can_handle(self, state: AnalysisState) -> bool:
        return 'missing_statistics' in state

    def _render(self, state: AnalysisState) -> None:
        pass


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
            for field in state.missing_values_field_analysis[section].keys():
                args = self.field_args.get(field, dict(fig_args={}, viz_args={}))
                fig_args = args['fig_args']
                viz_args = args['viz_args']
                field_analysis_tabs[field] = TargetFacetAnalysisVisualization(namespace=f'missing_values_field_analysis.{section}.{field}', fig_args=fig_args, viz_args=viz_args)
            field_analysis_tabs = TabLayout(facets=field_analysis_tabs)

            SimpleVerticalLinearLayout(
                facets=[
                    MarkdownSectionComponent(f'## Missing Values Fields Details - {title} number of missing values'),
                    field_analysis_tabs,
                ],
            ).render(state)

    class ParametersBuilder:
        __field_args = {}

        def with_field_details(self, field: str, parameters: TargetFacetAnalysisVisualization.ParametersBuilder):
            self.__field_args[field] = parameters.build()

        def build(self):
            return dict(field_args=self.__field_args)
