from __future__ import annotations

from typing import List, Type, Dict, Any

from ..backend.base import RenderingBackend
from ..backend.jupyter import SimpleJupyterRenderingToolsMixin
from ..base import AbstractAnalysis

ALL = '__all__'


class SequentialCompositeRenderer(RenderingBackend, SimpleJupyterRenderingToolsMixin):

    def render(self, model: Dict[str, Any]):
        for model, renderer in zip(model['component_models'], model['component_renderers']):
            renderer.render(model)


class CompositeAnalysis(AbstractAnalysis):

    def __init__(self,
                 children: List[AbstractAnalysis],
                 rendering_backend: Type[RenderingBackend] = SequentialCompositeRenderer,
                 **kwargs) -> None:
        super().__init__(children=children, rendering_backend=rendering_backend, **kwargs)

    def fit(self, **kwargs):
        component_models = []
        component_renderers = []
        for a in self.children:
            a.fit(**kwargs)
            component_models.append(a.model)
            component_renderers.append(a.rendering_backend)
        self.model = {
            'component_models': component_models,
            'component_renderers': component_renderers,
        }
