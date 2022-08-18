from __future__ import annotations

from typing import Union, List, Type, Dict, Any

import pandas as pd
from pandas import DataFrame

from ..backend.base import RenderingBackend
from ..backend.univariate import HistogramAnalysisRenderer, DatasetSummaryAnalysisRenderer
from ..base import AbstractAnalysis

ALL = '__all__'


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
        self.model = {
            'datasets': {},
            'types': None,
            'warnings': [],
        }
        for t, ds in self._datasets_as_map().items():
            if ds is not None:
                summary = ds.describe(include='all')
                if self.columns != ALL:
                    summary = summary[self.columns]
                summary = summary.T
                summary = summary.join(DataFrame({'dtypes': ds.dtypes}))
                summary = summary.sort_index()

                self.model['datasets'][t] = summary

        types = pd.DataFrame({t: self.model['datasets'][t]['dtypes'] for t in ['train_data', 'test_data', 'tuning_data'] if t in self.model['datasets']})
        self.model['types'] = types
        warnings = types.eq(types.iloc[:, 0], axis=0)
        types['warnings'] = warnings.all(axis=1).map({True: '', False: '⚠️'})
