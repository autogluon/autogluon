from __future__ import annotations

from typing import Dict, Any

from ..backend.base import RenderingBackend
from ..backend.jupyter import SimpleJupyterRenderingToolsMixin

ALL = '__all__'


class HistogramAnalysisRenderer(RenderingBackend, SimpleJupyterRenderingToolsMixin):

    def render(self, model: Dict[str, Any]):
        for t, ds in model['datasets'].items():
            self.render_text(f'Histogram for dataset: {t}', text_type='h2')
            self.render_histogram(ds, **model['figure_kwargs'])


class DatasetSummaryAnalysisRenderer(RenderingBackend, SimpleJupyterRenderingToolsMixin):

    def render(self, model: Dict[str, Any]):
        for t, summary in model['datasets'].items():
            self.render_text(f'Summary for dataset: {t}', text_type='h2')
            self.render_table(summary)

        if len(model['types'].columns) > 2:
            self.render_text(f'Types details', text_type='h2')
            self.render_table(model['types'])
