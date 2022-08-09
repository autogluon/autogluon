from __future__ import annotations

from typing import Union, List, Type, Dict, Any

from pandas import DataFrame

from ..backend.base import RenderingBackend
from ..backend.jupyter import SimpleJupyterRenderingToolsMixin
from ..base import AbstractAnalysis

ALL = '__all__'


class HistogramAnalysisRenderer(RenderingBackend, SimpleJupyterRenderingToolsMixin):

    def render(self, model: Dict[str, Any]):
        for t, ds in model['datasets'].items():
            self.render_text(f'Histogram for dataset: {t}', text_type='h2')
            self.render_histogram(ds, **model['figure_kwargs'])


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


class DatasetSummaryAnalysisRenderer(RenderingBackend, SimpleJupyterRenderingToolsMixin):

    def render(self, model: Dict[str, Any]):
        for t, summary in model['datasets'].items():
            self.render_text(f'Summary for dataset: {t}', text_type='h2')
            self.render_table(summary)


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
