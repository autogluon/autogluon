from typing import Union, Dict, Any

import ipywidgets as wgts
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import display
from scipy import stats

from . import AnalysisState
from ..util.types import map_raw_type_to_feature_type
from ..visualization import JupyterMixin, AbstractVisualization


class FeatureInteractionVisualization(AbstractVisualization, JupyterMixin):

    def __init__(self,
                 headers: bool = False,
                 namespace: str = None,
                 numeric_as_categorical_threshold=20,
                 fig_args: Union[None, Dict[str, Any]] = {},
                 chart_args: Union[None, Dict[str, Any]] = {},
                 render_only_idx: Union[None, int] = None,
                 **kwargs) -> None:
        super().__init__(namespace, **kwargs)
        self.headers = headers
        self.numeric_as_categorical_threshold = numeric_as_categorical_threshold
        self.fig_args = fig_args
        self.chart_args = chart_args
        self.render_only_idx = render_only_idx

    def can_handle(self, state: AnalysisState) -> bool:
        return self.all_keys_must_be_present(state, ['interactions', 'raw_type'])

    def _render(self, state: AnalysisState) -> None:
        for idx, i in enumerate(state.interactions):
            if self.render_only_idx is not None and self.render_only_idx != idx:
                continue
            ds = i['dataset']
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
                    **self.chart_args.get(idx, {}),
                }
                chart_args = {k: v for k, v in chart_args.items() if v is not None}
                if chart_type != 'kdeplot':
                    chart_args.pop('fill', None)
                if chart_type == 'barplot':
                    # Don't show ci ticks
                    chart_args['ci'] = None

                fig, ax = plt.subplots(**self.fig_args.get(idx, {}))

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

                self._get_sns_chart_method(chart_type)(ax=ax, data=data, **chart_args)

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
        return self.all_keys_must_be_present(state, ['correlations'])

    def _render(self, state: AnalysisState) -> None:
        for ds, corr in state.correlations.items():
            if len(state.correlations.train_data) <= 1:
                continue
            if state.correlations_focus_field is not None:
                focus_field_header = f'; focus: {state.correlations_focus_field} >= {state.correlations_focus_field_threshold}'
            else:
                focus_field_header = ''
            self.render_header_if_needed(state, f'{ds} - {state.correlations_method} correlation matrix{focus_field_header}')
            widgets = [w for w in ['correlations', 'significance_matrix'] if w in state]
            outs = [wgts.Output(scroll=False) for _ in widgets]
            tab = wgts.Tab(children=outs)
            for i, c in enumerate([w for w in widgets]):
                titles = {
                    'correlations': 'correlations',
                    'significance_matrix': 'significance matrix'
                }
                tab.set_title(i, titles[c])
            if len(widgets) > 1:
                display(tab)
            for widget, out in zip(widgets, outs):
                if len(widgets) > 1:
                    with out:
                        self.__render_internal(state, widget)
                else:
                    self.__render_internal(state, widget)

    def __render_internal(self, state, widget):
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
                    cbar_kws={"shrink": 0.5},
                    **args)
        plt.yticks(rotation=0)
        plt.show(fig)
