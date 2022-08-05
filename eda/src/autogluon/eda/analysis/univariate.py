from __future__ import annotations

from typing import TYPE_CHECKING, Union, List

from ..backend.base import RenderingBackend, EstimatorsBackend

if TYPE_CHECKING:
    from ..base import Analysis

from ..base import Facet

ALL = '__all__'


class Histogram(Facet):

    def __init__(self, datasets: Union[str, List[str]] = ALL, columns: Union[str, List[str]] = ALL, **kwargs):
        self.datasets = datasets
        self.columns = columns
        self._kwargs = kwargs

    def fit(self, ctx: Analysis, engine: EstimatorsBackend, **kwargs):
        self.model = {}
        for t, ds in ctx.datasets.items():
            cols = ds.columns
            if self.columns != ALL:
                cols = self.columns
            ds = ds[cols]
            self.model[t] = ds

    def render(self, engine: RenderingBackend, **kwargs):
        for t, ds in self.model.items():
            engine.render_text(f'Histogram for dataset: {t}', text_type='h2')
            engine.render_histogram(ds, **self._kwargs)


class Summary(Facet):

    def __init__(self, datasets: Union[str, List[str]] = ALL, columns: Union[str, List[str]] = ALL, **kwargs):
        self.datasets = datasets
        self.columns = columns

    def fit(self, ctx: Analysis, engine: EstimatorsBackend, **kwargs):
        self.model = {}
        for t, ds in ctx.datasets.items():
            if self.datasets == ALL or t in self.datasets:
                summary = ds.describe(include='all')
                if self.columns != ALL:
                    summary = summary[self.columns]
                summary = summary.T
                self.model[t] = summary

    def render(self, engine: RenderingBackend, **kwargs):
        for t, summary in self.model.items():
            engine.render_text(f'Summary for dataset: {t}', text_type='h2')
            engine.render_table(summary)
