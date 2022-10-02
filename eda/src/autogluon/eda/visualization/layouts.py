from typing import Union, List, Dict

from ipywidgets import HBox, Output, Layout, Tab

from . import JupyterMixin
from .. import AnalysisState
from ..visualization import AbstractVisualization


class SimpleVerticalLinearLayout(AbstractVisualization):

    def __init__(self,
                 facets: Union[AbstractVisualization, List[AbstractVisualization]],
                 namespace: str = None,
                 **kwargs) -> None:
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


class SimpleHorizontalLayout(SimpleVerticalLinearLayout, JupyterMixin):

    def _render(self, state: AnalysisState) -> None:
        outs = [Output() for i in range(len(self.facets))]
        for out, facet in zip(outs, self.facets):
            with out:
                facet.render(state)

        self.display_obj(HBox(outs, layout=Layout(flex='row wrap')))


class TabLayout(SimpleVerticalLinearLayout, JupyterMixin):

    def __init__(self,
                 facets: Dict[str, AbstractVisualization],
                 namespace: str = None,
                 **kwargs) -> None:

        self.facet_tab_names = list(facets.keys())
        super().__init__(list(facets.values()), namespace, **kwargs)

    def _render(self, state: AnalysisState) -> None:
        outs = [Output() for _ in self.facets]
        tab = Tab(children=outs, width=400)
        for i, name in enumerate(self.facet_tab_names):
            tab.set_title(i, name)
        self.display_obj(tab)
        for out, facet in zip(outs, self.facets):
            with out:
                facet.render(state)


class MarkdownSectionComponent(AbstractVisualization, JupyterMixin):

    def __init__(self,
                 markdown: str,
                 namespace: str = None,
                 **kwargs) -> None:
        super().__init__(namespace, **kwargs)
        self.markdown = markdown

    def can_handle(self, state: AnalysisState) -> bool:
        return True

    def _render(self, state: AnalysisState) -> None:
        self.render_markdown(self.markdown)
