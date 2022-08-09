import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from pandas import DataFrame


class SimpleJupyterRenderingToolsMixin:
    """
    High-level helpers for jupyter widgets rendering
    """

    def render_text(self, text, text_type=None):
        if text_type in [f'h{r}' for r in range(1, 7)]:
            display(HTML(f"<{text_type}>{text}</{text_type}>"))
        else:
            print(text)

    def render_table(self, df):
        display(df)

    def render_histogram(self, df: DataFrame, **kwargs):
        fig_params = kwargs.pop('fig_params', {})
        outs = [widgets.Output() for c in df.columns]
        tab = widgets.Tab(children=outs)
        for i, c in enumerate(df.columns):
            tab.set_title(i, c)
        display(tab)
        for c, out in zip(df.columns, outs):
            with out:
                fig, ax = plt.subplots(**fig_params)
                df.hist(ax=ax, column=c, **kwargs)
                plt.show(fig)
