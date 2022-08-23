from typing import Union, List

from autogluon.eda import AnalysisState
from autogluon.eda.visualization import AbstractVisualization


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
