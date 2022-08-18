from __future__ import annotations

from typing import Dict, Any

import matplotlib.pyplot as plt
import missingno as msno

from ..backend.base import RenderingBackend
from ..backend.jupyter import SimpleJupyterRenderingToolsMixin

ALL = '__all__'


class MissingStatisticsRenderer(RenderingBackend, SimpleJupyterRenderingToolsMixin):

    def render(self, model: Dict[str, Any]):
        kwargs = model['kwargs']
        for t, ds in model['datasets'].items():
            self.render_text(f'Missing statistics for dataset: {t}', text_type='h2')
            if model['hint'] is not None:
                self.render_text(model['hint'])
            if model['chart_type'] == 'matrix':
                msno.matrix(ds, **kwargs)
            elif model['chart_type'] == 'bar':
                msno.bar(ds, **kwargs)
            elif model['chart_type'] == 'heatmap':
                msno.heatmap(ds, **kwargs)
            elif model['chart_type'] == 'dendrogram':
                msno.dendrogram(ds, **kwargs)
            plt.show()


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
