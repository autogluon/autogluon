from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Union, List

import pandas as pd

if TYPE_CHECKING:
    from ..base import Analysis

from ..backend.base import RenderingBackend


class UnivariateAnalysis:

    def __init__(self, ctx, parent: Analysis = None, **kwargs):
        self.parent = parent
        self.params = kwargs
        self.ctx = ctx
        if parent and self.type:
            parent.facets.append(self)

    def hist(self, datasets: Union[str, List[str]] = '__all__', **kwargs) -> Histogram:
        return Histogram(deepcopy(self.ctx), datasets=datasets, parent=self.parent, **kwargs)

    def summary(self, datasets: Union[str, List[str]] = '__all__', **kwargs) -> Summary:
        return Summary(deepcopy(self.ctx), datasets=datasets, parent=self.parent, **kwargs)

    @property
    def type(self):
        ...

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def render(self, engine: RenderingBackend, **kwargs):
        raise NotImplementedError

    def __str__(self) -> str:
        return f'{self.type}: {self.params}'


class Histogram(UnivariateAnalysis):

    def __init__(self, ctx, parent: Analysis = None, datasets: Union[str, List[str]] = '__all__', **kwargs):
        self.datasets = datasets
        self._kwargs = kwargs
        super().__init__(ctx, parent, **kwargs)

    @property
    def type(self):
        return 'histogram'

    def fit(self):
        pass

    def render(self, engine, datasets: Union[str, List[str]] = '__all__', **kwargs):
        for t, ds in self.ctx['datasets'].items():
            cols = ds.columns
            if self.ctx['columns'] != '__all__':
                cols = self.ctx['columns']
            ds = ds[cols]
            engine.render_text(f'Histogram for dataset: {t}', text_type='h2')
            engine.render_histogram(ds, column=cols, **kwargs, **self._kwargs)


class Summary(UnivariateAnalysis):

    def __init__(self, ctx, parent: Analysis = None, datasets: Union[str, List[str]] = '__all__', **kwargs):
        self.datasets = datasets
        super().__init__(ctx, parent, **kwargs)

    @property
    def type(self):
        return 'summary'

    def fit(self):
        pass

    def render(self, engine, **kwargs):
        for t, ds in self.ctx['datasets'].items():
            if self.datasets == '__all__' or t in self.datasets:
                summary = ds.describe(include='all')
                if self.ctx['columns'] != '__all__':
                    summary = summary[self.ctx['columns']]
                summary = summary.T
                engine.render_text(f'Summary for dataset: {t}', text_type='h2')
                engine.render_table(summary)
