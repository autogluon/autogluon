from typing import Union, Dict, Any

import ipywidgets as wgts
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from pandas import DataFrame

from autogluon.common.features.types import *
from . import AnalysisState
from ..visualization import JupyterMixin, AbstractVisualization


class FeatureInteractionVisualization(AbstractVisualization, JupyterMixin):

    def __init__(self,
                 headers: bool = False,
                 namespace: str = None,
                 numeric_as_categorical_threshold=20,
                 fig_args: Union[None, Dict[str, Any]] = {},
                 **kwargs) -> None:
        super().__init__(namespace, **kwargs)
        self.headers = headers
        self.numeric_as_categorical_threshold = numeric_as_categorical_threshold
        self.fig_args = fig_args

    def can_handle(self, state: AnalysisState) -> bool:
        return self._all_keys_must_be_present(state, ['interactions', 'raw_types'])

    def _render(self, state: AnalysisState) -> None:
        for i in state.interactions:
            ds = i['dataset']
            df = i['data'].copy()
            x, y, hue = [i['features'].get(k, None) for k in ['x', 'y', 'hue']]
            x_type, y_type, hue_type = [self._map_raw_type_to_feature_type(var, state.raw_types[ds].get(var, None), df) for var in [x, y, hue]]
            chart_type = self._get_chart_type(x_type, y_type, hue_type)
            if chart_type is not None:
                chart_args = {
                    'x': x,
                    'y': y,
                    'hue': hue,
                    **self._kwargs,
                }
                if chart_type != 'kdeplot':
                    chart_args.pop('fill', None)
                if chart_type == 'barplot':
                    # Don't show ci ticks
                    chart_args['ci'] = None
                fig, ax = plt.subplots(**self.fig_args)

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
                self._get_sns_chart_method(chart_type)(ax=ax, data=data, **chart_args)
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
        }.get(chart_type)

    def _get_chart_type(self, x_type: str, y_type: str, hue_type: Union[None, str]) -> Union[None, str]:
        types = {
            ('numeric', None, None): 'kdeplot',
            ('category', None, None): 'countplot',
            (None, 'category', None): 'countplot',

            ('category', None, 'category'): 'countplot',
            (None, 'category', 'category'): 'countplot',

            ('numeric', None, 'category'): 'kdeplot',
            (None, 'numeric', 'category'): 'kdeplot',

            ('category', 'category', None): 'barplot',
            ('category', 'category', 'category'): 'barplot',

            ('category', 'numeric', None): 'boxplot',
            ('numeric', 'category', None): 'kdeplot',
            ('category', 'numeric', 'category'): 'boxplot',
            ('numeric', 'category', 'category'): 'kdeplot',

            ('numeric', 'numeric', None): 'scatterplot',
            ('numeric', 'numeric', 'category'): 'scatterplot',
        }
        return types.get((x_type, y_type, hue_type), None)

    def _map_raw_type_to_feature_type(self, col: str, raw_type: str, series: DataFrame) -> Union[None, str]:
        if col is None:
            return None
        elif series[col].nunique() <= self.numeric_as_categorical_threshold:
            return 'category'
        elif raw_type in [R_INT, R_FLOAT]:
            return 'numeric'
        elif raw_type in [R_OBJECT, R_CATEGORY, R_BOOL]:
            return 'category'
        else:
            return None


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
        return self._all_keys_must_be_present(state, ['correlations'])

    def _render(self, state: AnalysisState) -> None:
        for ds, corr in state.correlations.items():
            self.render_header_if_needed(state, f'{ds} - {state.correlations_method} correlation matrix')
            widgets = [w for w in ['correlations', 'significance_matrix'] if w in state]
            outs = [wgts.Output() for _ in widgets]
            tab = wgts.Tab(children=outs)
            for i, c in enumerate([w for w in widgets]):
                titles = {
                    'correlations': 'correlations',
                    'significance_matrix': 'significance matrix'
                }
                tab.set_title(i, titles[c])
            display(tab)
            for widget, out in zip(widgets, outs):
                with out:
                    fig, ax = plt.subplots(**self.fig_args)

                    if widget == 'significance_matrix':
                        args = {'center': 3, 'vmax': 5, 'cmap': 'Spectral', 'robust': True}
                    else:
                        args = {
                            'vmin': 0 if state.correlations_method == 'phik' else -1,
                            'vmax': 1, 'center': 0, 'cmap': 'Spectral'
                        }

                    sns.heatmap(state[widget].train_data,
                                annot=True,
                                ax=ax,
                                linewidths=.9,
                                linecolor='white',
                                fmt='.2f',
                                square=True,
                                **args)
                    plt.yticks(rotation=0)
                    plt.show(fig)
