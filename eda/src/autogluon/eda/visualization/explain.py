from abc import ABC, abstractmethod
from typing import Optional

import shap

from autogluon.eda import AnalysisState
from autogluon.eda.visualization.base import AbstractVisualization

__all__ = ["ExplainForcePlot", "ExplainWaterfallPlot"]

from autogluon.eda.visualization.jupyter import JupyterMixin


class _AbstractExplainPlot(AbstractVisualization, JupyterMixin, ABC):
    def __init__(self, display_rows: bool = False, namespace: Optional[str] = None, **kwargs) -> None:
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
        raise NotImplementedError  # pragma: no cover


class ExplainForcePlot(_AbstractExplainPlot):
    """
    Visualize the given SHAP values with an additive force layout

    Parameters
    ----------
    display_rows: bool, default = False
        if `True` then display the row before the explanation chart
    headers: bool, default = False
        if `True` then render headers
    namespace: str, default = None
        namespace to use; can be nested like `ns_a.ns_b.ns_c`

    Examples
    --------
    >>> import autogluon.eda.analysis as eda
    >>> import autogluon.eda.visualization as viz
    >>> import autogluon.eda.auto as auto
    >>>
    >>> rows_to_explain = ...  # DataFrame
    >>>
    >>> auto.analyze(
    >>>     train_data=..., model=...,
    >>>     anlz_facets=[
    >>>         eda.explain.ShapAnalysis(rows),
    >>>     ],
    >>>     viz_facets=[
    >>>         viz.explain.ExplainForcePlot(text_rotation=45, matplotlib=True),  # defaults used if not specified
    >>>     ]
    >>> )

    See Also
    --------
    :py:class:`~shap.KernelExplainer`
    :py:class:`~autogluon.eda.analysis.explain.ShapAnalysis`
    """

    def _render_internal(self, expected_value, shap_values, features, feature_names, **kwargs):
        _kwargs = {**dict(text_rotation=45, matplotlib=True), **kwargs}
        shap.force_plot(expected_value, shap_values, features, feature_names=feature_names, **_kwargs)


class ExplainWaterfallPlot(_AbstractExplainPlot):
    """
    Visualize the given SHAP values with a waterfall layout

    Parameters
    ----------
    display_rows: bool, default = False
        if `True` then display the row before the explanation chart
    headers: bool, default = False
        if `True` then render headers
    namespace: str, default = None
        namespace to use; can be nested like `ns_a.ns_b.ns_c`

    Examples
    --------
    >>> import autogluon.eda.analysis as eda
    >>> import autogluon.eda.visualization as viz
    >>> import autogluon.eda.auto as auto
    >>>
    >>> rows_to_explain = ...  # DataFrame
    >>>
    >>> auto.analyze(
    >>>     train_data=..., model=...,
    >>>     anlz_facets=[
    >>>         eda.explain.ShapAnalysis(rows_to_explain),
    >>>     ],
    >>>     viz_facets=[
    >>>         viz.explain.ExplainWaterfallPlot(),
    >>>     ]
    >>> )

    See Also
    --------
    :py:class:`~shap.KernelExplainer`
    :py:class:`~autogluon.eda.analysis.explain.ShapAnalysis`
    """

    def _render_internal(self, expected_value, shap_values, features, feature_names, **kwargs):
        shap.plots.waterfall(_ShapInput(expected_value, shap_values, features, feat_names=features.index), **kwargs)


class _ShapInput(object):
    def __init__(self, expectation, shap_values, features, feat_names):
        self.base_values = expectation
        self.values = shap_values
        self.display_data = features.values
        self.feature_names = list(feat_names)
