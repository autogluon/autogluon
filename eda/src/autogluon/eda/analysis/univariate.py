from __future__ import annotations

from typing import Union, List, Type, Dict, Any

from pandas import DataFrame

from ..backend.base import RenderingBackend
from ..backend.univariate import HistogramAnalysisRenderer, DatasetSummaryAnalysisRenderer, MissingStatisticsRenderer
from ..base import AbstractAnalysis

ALL = '__all__'


class MissingStatistics(AbstractAnalysis):

    def __init__(self,
                 sample=1000,
                 chart_type='matrix',
                 rendering_backend: Type[RenderingBackend] = MissingStatisticsRenderer,
                 **kwargs) -> None:
        super().__init__(rendering_backend=rendering_backend, **kwargs)
        self.sample = sample
        self.chart_type = chart_type
        self.data = None

    def fit(self, **kwargs):
        self.model = {
            'datasets': {},
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

        for t, ds in self._datasets_as_map().items():
            if ds is not None:
                if self.sample is not None:
                    if len(ds) > self.sample:
                        ds = ds.sample(self.sample)
                self.model['datasets'][t] = ds


class HistogramAnalysis(AbstractAnalysis):

    def __init__(self,
                 columns: Union[str, List[str]] = ALL,
                 rendering_backend: Type[RenderingBackend] = HistogramAnalysisRenderer,
                 figure_kwargs: Dict[str, Any] = {},
                 **kwargs) -> None:

        super().__init__(rendering_backend=rendering_backend, **kwargs)

        self.columns = columns
        self.figure_kwargs = figure_kwargs

    def fit(self, **kwargs):
        self.model = {
            'datasets': {},
            'figure_kwargs': self.figure_kwargs
        }
        for t, ds in self._datasets_as_map().items():
            if ds is not None:
                cols = ds.columns
                if self.columns != ALL:
                    cols = self.columns
                ds = ds[cols]
                self.model['datasets'][t] = ds


class DatasetSummaryAnalysis(AbstractAnalysis):

    def __init__(self,
                 train_data: Union[str, DataFrame] = None,
                 test_data: Union[str, DataFrame] = None,
                 tuning_data: Union[str, DataFrame] = None,
                 columns: Union[str, List[str]] = ALL,
                 rendering_backend: Type[RenderingBackend] = DatasetSummaryAnalysisRenderer,
                 children: List[AbstractAnalysis] = [],
                 **kwargs) -> None:

        super().__init__(
            train_data=train_data,
            test_data=test_data,
            tuning_data=tuning_data,
            rendering_backend=rendering_backend,
            children=children,
            **kwargs)

        self.columns = columns

    def fit(self, **kwargs):
        self.model = {'datasets': {}}
        for t, ds in self._datasets_as_map().items():
            if ds is not None:
                summary = ds.describe(include='all')
                if self.columns != ALL:
                    summary = summary[self.columns]
                summary = summary.T
                self.model['datasets'][t] = summary
