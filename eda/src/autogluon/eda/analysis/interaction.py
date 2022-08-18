from typing import Type

from autogluon.eda import AbstractAnalysis, RenderingBackend
from autogluon.eda.backend.interaction import TwoFeatureInteractionBoxplotRenderer, ThreeFeatureInteractionBoxplotRenderer


class TwoFeatureInteractionBoxplot(AbstractAnalysis):
    def __init__(self,
                 x,
                 y,
                 rendering_backend: Type[RenderingBackend] = TwoFeatureInteractionBoxplotRenderer,
                 **kwargs) -> None:
        super().__init__(rendering_backend=rendering_backend, **kwargs)
        self.y = y
        self.x = x

    def fit(self, **kwargs):
        self.model = {
            'x': self.x,
            'y': self.y,
            'kwargs': self._kwargs,
        }
        self._sample_and_set_model_datasets()
        self.model['datasets'] = {t: ds[[self.x, self.y]] for t, ds in self._get_datasets().items()}


class ThreeFeatureInteractionBoxplot(TwoFeatureInteractionBoxplot):
    def __init__(self,
                 hue,
                 rendering_backend: Type[RenderingBackend] = ThreeFeatureInteractionBoxplotRenderer,
                 **kwargs) -> None:
        super().__init__(rendering_backend=rendering_backend, **kwargs)
        self.hue = hue

    def fit(self, **kwargs):
        self.model = {
            'x': self.x,
            'y': self.y,
            'hue': self.hue,
            'kwargs': self._kwargs,
        }
        self._sample_and_set_model_datasets()
        self.model['datasets'] = {t: ds[[self.x, self.y, self.hue]] for t, ds in self._get_datasets().items()}
