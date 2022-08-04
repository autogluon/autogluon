from IPython.display import display, HTML
from pandas import DataFrame
import matplotlib.pyplot as plt
import ipywidgets as widgets

from autogluon.eda.backend.base import RenderingBackend


class SimpleJupyterBackend(RenderingBackend):

    def render_text(self, text, text_type=None):
        if text_type in [f'h{r}' for r in range(1, 7)]:
            display(HTML(f"<{text_type}>{text}</{text_type}>"))
        else:
            print(text)

    def render_table(self, df):
        display(df)

    def render_histogram(self, df: DataFrame, **kwargs):
        kwargs = kwargs.copy()
        kwargs.pop('column')
        fig_params = kwargs.pop('fig_params', {})

        outs = [widgets.Output() for c in df.columns]
        tab = widgets.Tab(children=outs)
        for i, c in enumerate(df.columns):
            tab.set_title(i, c)
        display(tab)
        print(kwargs)
        for c, out in zip(df.columns, outs):
            with out:
                fig, ax = plt.subplots(**fig_params)
                df.hist(ax=ax, column=c, **kwargs)
                plt.show(fig)

        # out = widgets.Output()
        # with out:
        #     fig, ax = plt.subplots()
        # df.hist(ax=ax, **kwargs)
        # plt.show(fig)

