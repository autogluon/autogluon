from typing import Any, Dict, Union

import matplotlib.pyplot as plt
import missingno as msno

from .base import AbstractVisualization
from .jupyter import JupyterMixin
from .. import AnalysisState

__all__ = ['MissingValues']


class MissingValues(AbstractVisualization, JupyterMixin):
    __OPERATIONS_MAPPING = {
        'matrix': msno.matrix,
        'bar': msno.bar,
        'heatmap': msno.heatmap,
        'dendrogram': msno.dendrogram,
    }

    def __init__(self,
                 graph_type: str,
                 headers: bool = False,
                 fig_args: Union[None, Dict[str, Any]] = {},
                 namespace: str = None,
                 **kwargs) -> None:
        super().__init__(namespace, **kwargs)
        self.graph_type = graph_type
        assert self.graph_type in self.__OPERATIONS_MAPPING, f'{self.graph_type} must be one of {self.__OPERATIONS_MAPPING.keys()}'
        self.headers = headers
        self.fig_args = fig_args

    def can_handle(self, state: AnalysisState) -> bool:
        return 'missing_statistics' in state

    def _render(self, state: AnalysisState) -> None:
        for ds, data in state.missing_statistics.items():
            self.render_header_if_needed(state, f'{ds} missing values analysis')
            widget = self.__OPERATIONS_MAPPING[self.graph_type]
            ax = widget(data.data, **self._kwargs)
            plt.show(ax)
