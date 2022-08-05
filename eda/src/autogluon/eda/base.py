from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Union, Dict

from pandas import DataFrame
from sklearn.compose import ColumnTransformer

from autogluon.eda.backend.base import RenderingBackend, EstimatorsBackend
from autogluon.eda.backend.jupyter import SimpleJupyterBackend


class Analysis:
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

    def __init__(self, datasets: Dict[str, DataFrame], target_col: str, rendering_backend: str = 'default', estimators_backend: str = 'autogluon') -> None:
        self.facets: List[Facet] = []
        self.datasets = datasets
        self.targe_col = target_col
        self.transform = None

        self.rendering_backend = {
            'default': SimpleJupyterBackend,
        }.get(rendering_backend, SimpleJupyterBackend)()

        # FIXME
        self.estimators_backend = {
            'autogluon': None,
        }.get(rendering_backend, None)

    def add_facets(self, facets: Union[Facet, List[Facet]]):
        if facets is Facet:
            facets = [facets]
        for f in facets:
            f.ctx = self
        self.facets += facets

    def apply_transform(self, transform: ColumnTransformer):
        self.transform = transform

    def fit(self, **kwargs):
        for f in self.facets:
            f.fit(ctx=self, engine=self.estimators_backend, **kwargs)

    def render(self, **kwargs):
        for f in self.facets:
            f.render(ctx=self, engine=self.rendering_backend, **kwargs)


class Facet(ABC):
    model = None

    @abstractmethod
    def fit(self, ctx: Analysis, engine: EstimatorsBackend, **kwargs):
        raise NotImplemented

    @abstractmethod
    def render(self, engine: RenderingBackend, **kwargs):
        raise NotImplemented
