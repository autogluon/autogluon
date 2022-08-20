from __future__ import annotations

from typing import Dict, Any

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .altair_base import AltairMixin
from ..backend.base import RenderingBackend
from ..backend.jupyter import SimpleJupyterRenderingToolsMixin

ALL = '__all__'


class TwoFeatureInteractionBoxplotRenderer(RenderingBackend, SimpleJupyterRenderingToolsMixin, AltairMixin):

    def render(self, model: Dict[str, Any]):
        source = []
        for t, ds in model['datasets'].items():
            ds = ds.copy()
            ds['dataset'] = t
            source.append(ds)
        source = pd.concat(source)

        kwargs = model['kwargs']

        x = f'{model["x"]}:{kwargs["x_type"]}' if 'x_type' in kwargs else model['x']
        y = f'{model["y"]}:{kwargs["y_type"]}' if 'y_type' in kwargs else model['y']

        boxplot_args = kwargs.get('boxplot_args', dict(size=25, outliers={'size': 10}, ticks=True))
        boxplot_properties = kwargs.get('boxplot_properties', {})

        bars = alt.Chart(source).mark_boxplot(**boxplot_args).encode(
            x=alt.X("dataset:N", title=None, axis=alt.Axis(labels=False, ticks=False), scale=alt.Scale(padding=1)),
            y=y,
            color=alt.Color('dataset:N'),
            column=alt.Column(x, header=alt.Header(orient='bottom'))
        ).properties(
            **boxplot_properties
        ).configure_facet(
            spacing=0
        ).configure_view(
            stroke=None
        )

        self.render_text(f'Interaction between {model["x"]} and {model["y"]}', text_type='h2')
        self.display_object(bars)


class ThreeFeatureInteractionBoxplotRenderer(RenderingBackend, SimpleJupyterRenderingToolsMixin, AltairMixin):

    def render(self, model: Dict[str, Any]):
        for t, ds in model['datasets'].items():
            source = ds.copy()

            kwargs = model['kwargs']

            x = f'{model["x"]}:{kwargs["x_type"]}' if 'x_type' in kwargs else model['x']
            y = f'{model["y"]}:{kwargs["y_type"]}' if 'y_type' in kwargs else model['y']
            hue = f'{model["hue"]}:{kwargs["hue_type"]}' if 'hue_type' in kwargs else model['hue']

            boxplot_args = kwargs.get('boxplot_args', dict(size=25, outliers={'size': 10}, ticks=True))
            boxplot_properties = kwargs.get('boxplot_properties', {})

            bars = alt.Chart(source).mark_boxplot(**boxplot_args).encode(
                x=alt.X(hue, title=None, axis=alt.Axis(labels=False, ticks=False), scale=alt.Scale(padding=1)),
                y=y,
                color=alt.Color(hue),
                column=alt.Column(x, header=alt.Header(orient='bottom'))
            ).properties(
                **boxplot_properties
            ).configure_facet(
                spacing=0
            ).configure_view(
                stroke=None
            )
            self.render_text(f'Interaction between {model["x"]}/{model["y"]}/{model["hue"]} in {t}', text_type='h2')
            self.display_object(bars)


class TwoFeatureInteractionBarchartRenderer(RenderingBackend, SimpleJupyterRenderingToolsMixin):

    def render(self, model: Dict[str, Any]):
        for t, source in model['datasets'].items():
            kwargs = model['kwargs'].copy()

            plot_kwargs = kwargs.pop('plot_kwargs', {})

            fig, ax = plt.subplots(**plot_kwargs)
            sns.barplot(
                ax=ax,
                x=model['x'],
                y=model['y'],
                data=source,
                **kwargs
            )

            self.render_text(f'Interaction between {model["x"]} and {model["y"]} in {t}', text_type='h2')
            plt.show(fig)


class TwoFeatureInteractionCountplotRenderer(RenderingBackend, SimpleJupyterRenderingToolsMixin):

    def render(self, model: Dict[str, Any]):
        for t, source in model['datasets'].items():
            kwargs = model['kwargs'].copy()

            plot_kwargs = kwargs.pop('plot_kwargs', {})

            fig, ax = plt.subplots(**plot_kwargs)
            sns.countplot(
                ax=ax,
                x=model['x'],
                hue=model['y'],
                data=source,
                **kwargs
            )

            self.render_text(f'Interaction between {model["x"]} and {model["y"]} in {t}', text_type='h2')
            plt.show(fig)


class TwoFeatureInteractionScatterplotRenderer(RenderingBackend, SimpleJupyterRenderingToolsMixin):

    def render(self, model: Dict[str, Any]):
        for t, source in model['datasets'].items():
            kwargs = model['kwargs'].copy()

            plot_kwargs = kwargs.pop('plot_kwargs', {})

            fig, ax = plt.subplots(**plot_kwargs)

            axes_args = dict(
                x=model['x'],
                y=model['y'],
            )
            if 'hue' in model:
                axes_args['hue'] = model['hue']

            sns.scatterplot(
                ax=ax,
                data=source,
                **axes_args,
                **kwargs
            )

            if 'hue' in model:
                self.render_text(f'Interaction between {model["x"]}/{model["y"]}/{model["hue"]} in {t}', text_type='h2')
            else:
                self.render_text(f'Interaction between {model["x"]} and {model["y"]} in {t}', text_type='h2')
            plt.show(fig)


class TwoFeatureInteractionKDEPlotRenderer(RenderingBackend, SimpleJupyterRenderingToolsMixin):

    def render(self, model: Dict[str, Any]):
        for t, source in model['datasets'].items():
            kwargs = model['kwargs'].copy()

            plot_kwargs = kwargs.pop('plot_kwargs', {})

            fig, ax = plt.subplots(**plot_kwargs)

            sns.kdeplot(
                ax=ax,
                x=model['x'],
                hue=model['y'],
                data=source,
                **kwargs
            )

            self.render_text(f'Interaction between {model["x"]} and {model["y"]} in {t}', text_type='h2')
            plt.show(fig)
