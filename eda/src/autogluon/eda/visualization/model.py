from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from yellowbrick.contrib.wrapper import REGRESSOR, ContribEstimator
from yellowbrick.regressor import residuals_plot

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


class _YellowbrickAutoGluonWrapper(ContribEstimator):
    _estimator_type = REGRESSOR

    def score(self, y_pred, y_true, **kwargs):
        # note: this is not conventional use of API: we pass y_pred since we already have predictions done
        return self.estimator.evaluate_predictions(y_pred, y_true)["r2"]

    def predict(self, y_pred, **kwargs):
        # note: this is not conventional use of API: we pass y_pred since we already have predictions done
        return y_pred


class RegressionEvaluation(AbstractVisualization, JupyterMixin):
    """
    This plot shows residuals on the vertical axis vs prediction on horizontal axis.

    This visualization depends on :py:class:`~autogluon.eda.analysis.model.AutoGluonModelEvaluator` analysis.

    Parameters
    ----------
    residuals_plot_mode: Optional[str], default = 'qoq'
        Additional plot to render to the right of the main plot. The supported values:
        - `qoq` (default) - Q-Q plot, which is a common way to check that residuals are normally distributed. If the residuals are normally distributed,
        then their quantiles when plotted against quantiles of normal distribution should form a straight line.
        - `hist` - display histogram that our error is normally distributed around zero, which also generally indicates a well fitted model
        - any other value - don't render additional details
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
        residuals_plot_mode: Optional[str] = "qoq",
        fig_args: Optional[Dict[str, Any]] = None,
        headers: bool = False,
        namespace: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(namespace, **kwargs)
        self.headers = headers
        self.residuals_analysis_mode = residuals_plot_mode

        if fig_args is None:
            fig_args = {}
        fig_args = {**{"figsize": (12, 6)}, **fig_args}
        self.fig_args = fig_args

    def can_handle(self, state: AnalysisState) -> bool:
        return "model_evaluation" in state and state.model_evaluation.problem_type == REGRESSION

    def _get_plot_mode(self):
        res_plot_kwargs = {
            "hist": dict(hist=True, qqplot=False),
            "qoq": dict(hist=False, qqplot=True),
        }.get(
            self.residuals_analysis_mode, dict(hist=False, qqplot=False)  # type: ignore
        )
        return res_plot_kwargs

    def _render(self, state: AnalysisState) -> None:
        self.render_header_if_needed(state, "Prediction vs Target")
        res_plot_kwargs = self._get_plot_mode()
        fig, ax = plt.subplots(**self.fig_args)
        y_pred_train, y_true_train, y_pred_test, y_true_test = RegressionEvaluation._repack_parameters(
            state.model_evaluation
        )
        residuals_plot(
            _YellowbrickAutoGluonWrapper(state.model),
            y_pred_train,
            y_true_train,
            y_pred_test,
            y_true_test,
            show=False,
            ax=ax,
            **res_plot_kwargs,
        )
        plt.show(fig)

    @staticmethod
    def _repack_parameters(ev):
        y_pred_train = ev.y_pred_train if "y_pred_train" in ev else ev.y_pred_val
        y_true_train = ev.y_true_train if "y_pred_train" in ev else ev.y_true_val
        y_pred_test = ev.y_pred_test if "y_true_test" in ev else (ev.y_pred_val if "y_pred_train" in ev else None)
        y_true_test = ev.y_true_test if "y_true_test" in ev else (ev.y_true_val if "y_pred_train" in ev else None)
        return y_pred_train, y_true_train, y_pred_test, y_true_test


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
