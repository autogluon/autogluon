from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any

from pandas import DataFrame
from sklearn.compose import ColumnTransformer

from autogluon.eda.backend.base import RenderingBackend, EstimatorsBackend
from autogluon.eda.backend.jupyter import SimpleJupyterBackend


class Dataset:
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class Analysis:

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
        }.get(estimators_backend, None)

    def add_facets(self, facets: Union[Facet, List[Facet]]):
        """
        Adds EDA primitives for this analysis
        """
        if facets is Facet:
            facets = [facets]
        for f in facets:
            f.ctx = self
        self.facets += facets

    def apply_transform(self, transform: ColumnTransformer):
        """
        Apply transformation to the data copy before performing analysis.
        """
        self.transform = transform

    def fit(self, **kwargs):
        """
        Fit composing EDA primitives.
        """
        for f in self.facets:
            f.fit(ctx=self, engine=self.estimators_backend, **kwargs)

    def render(self, **kwargs):
        """
        Fit composing EDA primitives.
        """
        for f in self.facets:
            f.render(ctx=self, engine=self.rendering_backend, **kwargs)


class Facet(ABC):
    # model storing fitted metadata for rendering
    model: Dict[str, Any] = None

    @abstractmethod
    def fit(self, ctx: Analysis, engine: EstimatorsBackend, **kwargs) -> None:
        """
        Fits primitive and populates `model` property for further rendering.
        This method should only update model and should be UI-agnostic (pass only
        data structures).

        Parameters
        ----------
        ctx: Analysis
            parent context. Things shared between primitives are passed
            via parent Analysis object (i.e. datasets, target_col, etc).

        engine: EstimatorsBackend
            estimators backend. Required for facets dependent on estimators


        """
        raise NotImplemented

    @abstractmethod
    def render(self, engine: RenderingBackend, **kwargs) -> None:
        """
        Renders `model` fitted by :func:`fit` using provided `engine`.

        Parameters
        ----------
        engine
        kwargs


        """
        raise NotImplemented
