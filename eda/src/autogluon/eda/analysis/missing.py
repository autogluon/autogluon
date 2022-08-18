from __future__ import annotations

from typing import Type

from ..backend.base import RenderingBackend
from ..backend.missing import MissingStatisticsRenderer
from ..base import AbstractAnalysis

ALL = '__all__'


# TODO: add composite complex analysis
# TODO: add analysis options for time series (freq)

class MissingStatistics(AbstractAnalysis):

    def __init__(self,
                 chart_type='matrix',
                 rendering_backend: Type[RenderingBackend] = MissingStatisticsRenderer,
                 **kwargs) -> None:
        super().__init__(rendering_backend=rendering_backend, **kwargs)
        self.chart_type = chart_type
        self.data = None

    def fit(self, **kwargs):
        self.model = {
            'chart_type': self.chart_type,
            'kwargs': self._kwargs,
            'hint': {
                'matrix': 'nullity matrix is a data-dense display which lets you quickly visually pick out patterns in data completion',
                'bar': 'simple visualization of nullity by column',
                'heatmap': 'correlation heatmap measures nullity correlation: how strongly the presence or absence of one variable affects the presence of another',
                'dendrogram': 'The dendrogram uses a hierarchical clustering algorithm to bin variables against one another by '
                              'their nullity correlation (measured in terms of binary distance). The more monotone the set of variables, '
                              'the closer their total distance is to zero, and the closer their average distance (the y-axis) is to zero.',
            }[self.chart_type]
        }

        self._sample_and_set_model_datasets()
