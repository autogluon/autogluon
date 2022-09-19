from typing import Union, List

from . import JupyterMixin
from .. import AnalysisState
from ..visualization import AbstractVisualization


class SimpleLinearLayout:

    def __init__(self,
                 state: AnalysisState,
                 facets: Union[AbstractVisualization, List[AbstractVisualization]],
                 **kwargs):
        self.state = state
        if not isinstance(facets, list):
            facets = [facets]
        self.facets = facets
        self._kwargs = kwargs

    def render(self):
        for facet in self.facets:
            facet.render(self.state)


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
