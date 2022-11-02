from typing import Union, List, Dict

from IPython.display import display, Markdown
from ipywidgets import HBox, Output, Layout, Tab

from .base import AbstractVisualization
from .. import AnalysisState

__all__ = ["MarkdownSectionComponent", "SimpleVerticalLinearLayout", "SimpleHorizontalLayout", "TabLayout"]


class SimpleVerticalLinearLayout(AbstractVisualization):
    """
    Renders facets in a sequential order (facets will appear in a vertical layout).
    """

    def __init__(
        self, facets: Union[AbstractVisualization, List[AbstractVisualization]], namespace: str = None, **kwargs
    ) -> None:
        super().__init__(namespace, **kwargs)
        if not isinstance(facets, list):
            facets = [facets]
        self.facets = facets
        self._kwargs = kwargs

    def can_handle(self, state: AnalysisState) -> bool:
        return True

    def _render(self, state: AnalysisState) -> None:
        for facet in self.facets:
            facet.render(state)


class SimpleHorizontalLayout(SimpleVerticalLinearLayout):
    """
    Render components horizontally using `HBox` widget.
    See `HBox widget <https://ipywidgets.readthedocs.io/en/stable/examples/Widget%20List.html#HBox>`_ documentation for details.
    """

    def _render(self, state: AnalysisState) -> None:
        outs = [Output() for _ in range(len(self.facets))]
        for out, facet in zip(outs, self.facets):
            with out:
                facet.render(state)

        display(HBox(outs, layout=Layout(flex="row wrap")))


class TabLayout(SimpleVerticalLinearLayout):
    """
    Render components using `Tab` widget.
    See `HBox widget <https://ipywidgets.readthedocs.io/en/stable/examples/Widget%20List.html#Tabs>`_ documentation for details.

    """

    def __init__(self, facets: Dict[str, AbstractVisualization], namespace: str = None, **kwargs) -> None:

        self.facet_tab_names = list(facets.keys())
        super().__init__(list(facets.values()), namespace, **kwargs)

    def _render(self, state: AnalysisState) -> None:
        outs = [Output() for _ in self.facets]
        tab = Tab(children=outs, width=400)
        for i, name in enumerate(self.facet_tab_names):
            tab.set_title(i, name)
        display(tab)
        for out, facet in zip(outs, self.facets):
            with out:
                facet.render(state)


class MarkdownSectionComponent(AbstractVisualization):
    """
    Render provided string as a Markdown cell.
    See `Jupyter Markdown cell <https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Working%20With%20Markdown%20Cells.html>`_
    documentation for details.
    """

    def __init__(self, markdown: str, namespace: str = None, **kwargs) -> None:
        super().__init__(namespace, **kwargs)
        self.markdown = markdown

    def can_handle(self, state: AnalysisState) -> bool:
        return True

    def _render(self, state: AnalysisState) -> None:
        display(Markdown(self.markdown))
