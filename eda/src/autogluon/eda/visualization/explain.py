from abc import ABC, abstractmethod
from typing import Optional

import shap

from autogluon.eda import AnalysisState
from autogluon.eda.visualization.base import AbstractVisualization

__all__ = ["ExplainForcePlot", "ExplainWaterfallPlot"]

from autogluon.eda.visualization.jupyter import JupyterMixin


class _AbstractExplainPlot(AbstractVisualization, JupyterMixin, ABC):
    def __init__(self, display_rows: bool = True, namespace: Optional[str] = None, **kwargs) -> None:
        super().__init__(namespace, **kwargs)
        self.display_rows = display_rows

    def can_handle(self, state: AnalysisState) -> bool:
        return self.all_keys_must_be_present(state, "explain") and self.all_keys_must_be_present(
            state.explain, "shapley"
        )

    def _render(self, state: AnalysisState) -> None:
        for s in state.explain.shapley:
            if self.display_rows:
                self.display_obj(s.row)
            self._render_internal(
                s.expected_value, s.shap_values, s.features, feature_names=s.feature_names, **self._kwargs
            )

    @abstractmethod
    def _render_internal(self, expected_value, shap_values, features, feature_names, **kwargs):
        raise NotImplementedError


class ExplainForcePlot(_AbstractExplainPlot):
    def _render_internal(self, expected_value, shap_values, features, feature_names, **kwargs):
        _kwargs = {**dict(text_rotation=45, matplotlib=True), **kwargs}
        shap.force_plot(expected_value, shap_values, features, feature_names=feature_names, **_kwargs)


class ExplainWaterfallPlot(_AbstractExplainPlot):
    def _render_internal(self, expected_value, shap_values, features, feature_names, **kwargs):
        shap.plots._waterfall.waterfall_legacy(
            expected_value, shap_values, features, feature_names=feature_names, **kwargs
        )
