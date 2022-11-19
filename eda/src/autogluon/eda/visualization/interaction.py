from typing import Union, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy as hc

from .base import AbstractVisualization
from .jupyter import JupyterMixin
from ..state import AnalysisState, StateCheckMixin
from ..util.types import map_raw_type_to_feature_type

__all__ = ['CorrelationVisualization', 'CorrelationSignificanceVisualization', 'FeatureDistanceAnalysisVisualization', 'FeatureInteractionVisualization']


class FeatureInteractionVisualization(AbstractVisualization, JupyterMixin):

    def __init__(self,
                 key: str,
                 headers: bool = False,
                 namespace: str = None,
                 numeric_as_categorical_threshold=20,
                 fig_args: Union[None, Dict[str, Any]] = {},
                 chart_args: Union[None, Dict[str, Any]] = {},
                 **kwargs) -> None:
        super().__init__(namespace, **kwargs)
        self.key = key
        self.headers = headers
        self.numeric_as_categorical_threshold = numeric_as_categorical_threshold
        self.fig_args = fig_args
        self.chart_args = chart_args

    def can_handle(self, state: AnalysisState) -> bool:
        return self.all_keys_must_be_present(state, 'interactions', 'raw_type')

    def _render(self, state: AnalysisState) -> None:
        for ds in state.interactions.keys():
            if self.key not in state.interactions[ds]:
                continue
            i = state.interactions[ds][self.key]
            df = i['data'].copy()
            x, y, hue = [i['features'].get(k, None) for k in ['x', 'y', 'hue']]
            x_type, y_type, hue_type = [map_raw_type_to_feature_type(var, state.raw_type[ds].get(var, None), df, self.numeric_as_categorical_threshold)
                                        for var in [x, y, hue]]

            # swap y <-> hue when category vs category to enable charting
            if (x_type is not None) and y_type == 'category' and hue is None:
                hue, hue_type = y, y_type
                y, y_type = None, None

            chart_type = self._get_chart_type(x_type, y_type, hue_type)
            if chart_type is not None:
                chart_args = {
                    'x': x,
                    'y': y,
                    'hue': hue,
                    **self.chart_args.get(self.key, {}),
                }
                chart_args = {k: v for k, v in chart_args.items() if v is not None}
                if chart_type != 'kdeplot':
                    chart_args.pop('fill', None)
                if chart_type == 'barplot':
                    # Don't show ci ticks
                    chart_args['ci'] = None

                fig, ax = plt.subplots(**self.fig_args.get(self.key, {}))

                # convert to categoricals for plots
                for col, typ in zip([x, y, hue], [x_type, y_type, hue_type]):
                    if typ == 'category':
                        df[col] = df[col].astype('object')

                data = df
                single_var = False
                if y is None and hue is None:
                    single_var = True
                    if x_type == 'numeric':
                        data = df[x]
                        chart_args.pop('x')
                elif x is None and hue is None:
                    single_var = True
                    if y_type == 'numeric':
                        data = df[y]
                        chart_args.pop('y')

                # Handling fitted distributions if present
                dists = None
                if (chart_type == 'histplot') and ('distributions_fit' in state) and ((x_type, y_type, hue_type) == ('numeric', None, None)):
                    chart_args['stat'] = 'density'
                    dists = state.distributions_fit[ds][x]

                chart = self._get_sns_chart_method(chart_type)(ax=ax, data=data, **chart_args)
                if chart_type in ('countplot', 'barplot', 'boxplot'):
                    plt.setp(chart.get_xticklabels(), rotation=90)

                if dists is not None:
                    x_min, x_max = ax.get_xlim()
                    xs = np.linspace(x_min, x_max, 200)
                    for dist, v in dists.items():
                        _dist = getattr(stats, dist)
                        ax.plot(xs, _dist.pdf(xs, *dists[dist]['param']), ls='--', lw=0.7, label=f'{dist}: pvalue {dists[dist]["pvalue"]:.2f}')
                    ax.set_xlim(x_min, x_max)  # set the limits back to the ones of the distplot
                    plt.legend()

                if chart_type == 'countplot':
                    for container in ax.containers:
                        ax.bar_label(container)

                if self.headers:
                    features = '/'.join([i['features'][k] for k in ['x', 'y', 'hue'] if k in i['features']])
                    prefix = '' if single_var else 'Feature interaction between '
                    self.render_header_if_needed(state, f'{prefix}{features} in {ds}')
                plt.show(fig)

    def _get_sns_chart_method(self, chart_type):
        return {
            'countplot': sns.countplot,
            'barplot': sns.barplot,
            'boxplot': sns.boxplot,
            'kdeplot': sns.kdeplot,
            'scatterplot': sns.scatterplot,
            'regplot': sns.regplot,
            'histplot': sns.histplot,
        }.get(chart_type)

    def _get_chart_type(self, x_type: str, y_type: str, hue_type: Union[None, str]) -> Union[None, str]:
        types = {
            ('numeric', None, None): 'histplot',
            ('category', None, None): 'countplot',
            (None, 'category', None): 'countplot',

            ('category', None, 'category'): 'countplot',
            (None, 'category', 'category'): 'countplot',

            ('numeric', None, 'category'): 'histplot',
            (None, 'numeric', 'category'): 'histplot',

            ('category', 'category', None): 'barplot',
            ('category', 'category', 'category'): 'barplot',

            ('category', 'numeric', None): 'boxplot',
            ('numeric', 'category', None): 'kdeplot',
            ('category', 'numeric', 'category'): 'boxplot',
            ('numeric', 'category', 'category'): 'kdeplot',

            ('numeric', 'numeric', None): 'regplot',
            ('numeric', 'numeric', 'category'): 'scatterplot',
        }
        return types.get((x_type, y_type, hue_type), None)


class CorrelationVisualization(AbstractVisualization, JupyterMixin):

    def __init__(self,
                 headers: bool = False,
                 namespace: str = None,
                 fig_args: Union[None, Dict[str, Any]] = {},
                 **kwargs) -> None:
        super().__init__(namespace, **kwargs)
        self.headers = headers
        self.fig_args = fig_args

    def can_handle(self, state: AnalysisState) -> bool:
        return 'correlations' in state

    def _render(self, state: AnalysisState) -> None:
        for ds, corr in state.correlations.items():
            # Don't render single cell
            if len(state.correlations[ds]) <= 1:
                continue

            if state.correlations_focus_field is not None:
                focus_field_header = f'; focus: {state.correlations_focus_field} >= {state.correlations_focus_field_threshold}'
            else:
                focus_field_header = ''
            self.render_header_if_needed(state, f'{ds} - {state.correlations_method} correlation matrix{focus_field_header}')

            fig, ax = plt.subplots(**self.fig_args)
            args = {
                'vmin': 0 if state.correlations_method == 'phik' else -1,
                'vmax': 1, 'center': 0, 'cmap': 'Spectral'
            }
            sns.heatmap(corr,
                        annot=True,
                        ax=ax,
                        linewidths=.9,
                        linecolor='white',
                        fmt='.2f',
                        square=True,
                        cbar_kws={"shrink": 0.5},
                        **args)
            plt.yticks(rotation=0)
            plt.show(fig)


class CorrelationSignificanceVisualization(AbstractVisualization, JupyterMixin):

    def __init__(self,
                 headers: bool = False,
                 namespace: str = None,
                 fig_args: Union[None, Dict[str, Any]] = {},
                 **kwargs) -> None:
        super().__init__(namespace, **kwargs)
        self.headers = headers
        self.fig_args = fig_args

    def can_handle(self, state: AnalysisState) -> bool:
        return 'significance_matrix' in state

    def _render(self, state: AnalysisState) -> None:
        for ds, corr in state.correlations.items():
            # Don't render single cell
            if len(state.significance_matrix[ds]) <= 1:
                continue

            if state.correlations_focus_field is not None:
                focus_field_header = f'; focus: {state.correlations_focus_field} >= {state.correlations_focus_field_threshold}'
            else:
                focus_field_header = ''
            self.render_header_if_needed(state, f'{ds} - {state.correlations_method} correlation matrix significance{focus_field_header}')

            fig, ax = plt.subplots(**self.fig_args)
            args = {'center': 3, 'vmax': 5, 'cmap': 'Spectral', 'robust': True}

            sns.heatmap(corr,
                        annot=True,
                        ax=ax,
                        linewidths=.9,
                        linecolor='white',
                        fmt='.2f',
                        square=True,
                        cbar_kws={"shrink": 0.5},
                        **args)
            plt.yticks(rotation=0)
            plt.show(fig)


class FeatureDistanceAnalysisVisualization(AbstractVisualization, StateCheckMixin, JupyterMixin):
    def __init__(self,
                 namespace: str = None,
                 fig_args: Union[None, Dict[str, Any]] = {},
                 **kwargs) -> None:
        super().__init__(namespace, **kwargs)
        self.fig_args = fig_args

    def can_handle(self, state: AnalysisState) -> bool:
        return self.all_keys_must_be_present(state, 'feature_distance')

    def _render(self, state: AnalysisState) -> None:
        fig, ax = plt.subplots(**self.fig_args)
        default_args = dict(
            orientation='left'
        )
        ax.grid(False)
        hc.dendrogram(ax=ax, Z=state.feature_distance.linkage, labels=state.feature_distance.columns, **{**default_args, **self._kwargs})
        plt.show(fig)
        if len(state.feature_distance.near_duplicates) > 0:
            message = f'**The following feature groups are considered as near-duplicates**:\n\n' \
                      f'Distance threshold: <= `{state.feature_distance.near_duplicates_threshold}`. ' \
                      f'Consider keeping only some of the columns within each group:\n'
            for group in state.feature_distance.near_duplicates:
                message += f'\n - `{"`, `".join(sorted(group["nodes"]))}` - distance `{group["distance"]:.2f}`'
            self.render_markdown(message)

class NonparametricSignificanceVisualization(AbstractVisualization, JupyterMixin):

    def __init__(self,
                 headers: bool = False,
                 namespace: str = None,
                 fig_args: Union[None, Dict[str, Any]] = {},
                 pvalue_cmap: str = 'rocket',
                 **kwargs) -> None:
        super().__init__(namespace, **kwargs)
        self.headers = headers
        self.fig_args = fig_args
        self.cmap = pvalue_cmap

    def can_handle(self, state: AnalysisState) -> bool:
        return 'association_pvalue_matrix' in state

    def _render(self, state: AnalysisState) -> None:
        if self.headers:
            self.render_header_if_needed(state, 'P-values for non-parametric association tests')

        fig, ax = plt.subplots(**self.fig_args)
        args = {'vmin': 0, 'vmax': 1, 'cmap': self.cmap}
        corr = state.association_pvalue_matrix

        sns.heatmap(corr,
                    annot=True,
                    ax=ax,
                    linewidths=.9,
                    linecolor='white',
                    fmt='.2f',
                    square=True,
                    # cbar_kws={"shrink": 0.5},
                    **args)
        self.display_obj(fig)
