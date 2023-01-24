from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from autogluon.core.constants import REGRESSION

from ..state import AnalysisState
from .base import AbstractVisualization
from .jupyter import JupyterMixin

__all__ = ["ConfusionMatrix", "FeatureImportance", "RegressionEvaluation", "ModelLeaderboard"]


class ConfusionMatrix(AbstractVisualization, JupyterMixin):
    """
    Render confusion matrix for binary/multiclass classificator.

    This visualization depends on :py:class:`~autogluon.eda.analysis.model.AutoGluonModelEvaluator` analysis.

    Parameters
    ----------
    headers: bool, default = False
        if `True` then render headers
    namespace: str, default = None
        namespace to use; can be nested like `ns_a.ns_b.ns_c`
    fig_args: Optional[Dict[str, Any]] = None,
        kwargs to pass into chart figure

    Examples
    --------
    >>> import autogluon.eda.analysis as eda
    >>> import autogluon.eda.visualization as viz
    >>> import autogluon.eda.auto as auto
    >>>
    >>> df_train = ...
    >>> df_test = ...
    >>> predictor = ...
    >>>
    >>> auto.analyze(model=predictor, val_data=df_test, anlz_facets=[
    >>>     eda.model.AutoGluonModelEvaluator(),
    >>> ], viz_facets=[
    >>>     viz.model.ConfusionMatrix(fig_args=dict(figsize=(3,3)), annot_kws={"size": 12}),
    >>> ])

    See Also
    --------
    :py:class:`~autogluon.eda.analysis.model.AutoGluonModelEvaluator`
    """

    def __init__(
        self,
        fig_args: Optional[Dict[str, Any]] = None,
        headers: bool = False,
        namespace: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(namespace, **kwargs)
        self.headers = headers

        if fig_args is None:
            fig_args = {}
        self.fig_args = fig_args

    def can_handle(self, state: AnalysisState) -> bool:
        return "model_evaluation" in state and "confusion_matrix" in state.model_evaluation

    def _render(self, state: AnalysisState) -> None:
        self.render_header_if_needed(state, "Confusion Matrix")
        labels = state.model_evaluation.labels
        cm = pd.DataFrame(state.model_evaluation.confusion_matrix, columns=labels, index=labels)
        cm.index.name = "Actual"
        cm.columns.name = "Predicted"
        normalized = state.model_evaluation.confusion_matrix_normalized
        fmt = ",.2%" if normalized else "d"

        cells_num = len(cm)
        fig_args = self.fig_args.copy()
        if "figsize" not in fig_args:
            fig_args["figsize"] = (cells_num, cells_num)

        fig, ax = plt.subplots(**fig_args)
        sns.heatmap(
            cm,
            ax=ax,
            cmap="Blues",
            annot=True,
            linewidths=0.5,
            linecolor="lightgrey",
            fmt=fmt,
            cbar=False,
            **self._kwargs,
        )
        plt.show(fig)


class RegressionEvaluation(AbstractVisualization, JupyterMixin):
    """
    Render predictions vs ground truth chart for regressor.

    This visualization depends on :py:class:`~autogluon.eda.analysis.model.AutoGluonModelEvaluator` analysis.

    Parameters
    ----------
    headers: bool, default = False
        if `True` then render headers
    namespace: str, default = None
        namespace to use; can be nested like `ns_a.ns_b.ns_c`
    fig_args: Optional[Dict[str, Any]] = None,
        kwargs to pass into chart figure

    Examples
    --------
    >>> import autogluon.eda.analysis as eda
    >>> import autogluon.eda.visualization as viz
    >>> import autogluon.eda.auto as auto
    >>>
    >>> df_train = ...
    >>> df_test = ...
    >>> predictor = ...
    >>>
    >>> auto.analyze(model=predictor, val_data=df_test, anlz_facets=[
    >>>     eda.model.AutoGluonModelEvaluator(),
    >>> ], viz_facets=[
    >>>     viz.model.RegressionEvaluation(fig_args=dict(figsize=(6,6)), marker='o', scatter_kws={'s':5}),
    >>> ])

    See Also
    --------
    :py:class:`~autogluon.eda.analysis.model.AutoGluonModelEvaluator`
    """

    def __init__(
        self,
        fig_args: Optional[Dict[str, Any]] = None,
        headers: bool = False,
        namespace: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(namespace, **kwargs)
        self.headers = headers

        if fig_args is None:
            fig_args = {}
        self.fig_args = fig_args

    def can_handle(self, state: AnalysisState) -> bool:
        return "model_evaluation" in state and state.model_evaluation.problem_type == REGRESSION

    def _render(self, state: AnalysisState) -> None:
        self.render_header_if_needed(state, "Prediction vs Target")
        data = pd.DataFrame({"y_true": state.model_evaluation.y_true, "y_pred": state.model_evaluation.y_pred})
        fig, ax = plt.subplots(**self.fig_args)
        sns.regplot(ax=ax, data=data, x="y_true", y="y_pred", **self._kwargs)
        plt.show(fig)


class FeatureImportance(AbstractVisualization, JupyterMixin):
    """
    Render feature importance for the model.

    This visualization depends on :py:class:`~autogluon.eda.analysis.model.AutoGluonModelEvaluator` analysis.

    Parameters
    ----------
    show_barplots: bool, default = False
        render features barplots if True
    headers: bool, default = False
        if `True` then render headers
    namespace: str, default = None
        namespace to use; can be nested like `ns_a.ns_b.ns_c`
    fig_args: Optional[Dict[str, Any]] = None,
        kwargs to pass into chart figure

    Examples
    --------
    >>> import autogluon.eda.analysis as eda
    >>> import autogluon.eda.visualization as viz
    >>> import autogluon.eda.auto as auto
    >>>
    >>> df_train = ...
    >>> df_test = ...
    >>> predictor = ...
    >>>
    >>> auto.analyze(model=predictor, val_data=df_test, anlz_facets=[
    >>>     eda.model.AutoGluonModelEvaluator(),
    >>> ], viz_facets=[
    >>>     viz.model.FeatureImportance(show_barplots=True)
    >>> ])

    See Also
    --------
    :py:class:`~autogluon.eda.analysis.model.AutoGluonModelEvaluator`
    """

    def __init__(
        self,
        show_barplots: bool = False,
        fig_args: Optional[Dict[str, Any]] = None,
        headers: bool = False,
        namespace: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(namespace, **kwargs)
        self.headers = headers

        if fig_args is None:
            fig_args = {}
        self.fig_args = fig_args

        self.show_barplots = show_barplots

    def can_handle(self, state: AnalysisState) -> bool:
        return "model_evaluation" in state and "importance" in state.model_evaluation

    def _render(self, state: AnalysisState) -> None:
        self.render_header_if_needed(state, "Feature Importance")
        importance = state.model_evaluation.importance
        with pd.option_context("display.max_rows", 100 if len(importance) <= 100 else 20):
            self.display_obj(importance)
        if self.show_barplots:
            fig_args = self.fig_args.copy()
            if "figsize" not in fig_args:
                fig_args["figsize"] = (12, len(importance) / 4)

            fig, ax = plt.subplots(**fig_args)
            sns.barplot(ax=ax, data=importance.reset_index(), y="index", x="importance", **self._kwargs)
            plt.show(fig)


class ModelLeaderboard(AbstractVisualization, JupyterMixin):
    """
    Render model leaderboard for trained model ensemble.

    Parameters
    ----------
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
    >>> df_train = ...
    >>> df_test = ...
    >>> predictor = ...
    >>>
    >>> auto.analyze(model=predictor, val_data=df_test, anlz_facets=[
    >>>     eda.model.AutoGluonModelEvaluator(),
    >>> ], viz_facets=[
    >>>     viz.model.ModelLeaderboard(),
    >>> ])

    See Also
    --------
    :py:class:`~autogluon.eda.analysis.model.AutoGluonModelEvaluator`
    """

    def __init__(self, namespace: Optional[str] = None, headers: bool = False, **kwargs) -> None:
        super().__init__(namespace, **kwargs)
        self.headers = headers

    def can_handle(self, state: AnalysisState) -> bool:
        return "model_evaluation" in state and "leaderboard" in state.model_evaluation

    def _render(self, state: AnalysisState) -> None:
        self.render_header_if_needed(state, "Model Leaderboard")
        df = state.model_evaluation.leaderboard
        with pd.option_context("display.max_rows", 100 if len(df) <= 100 else 20):
            self.display_obj(df)
