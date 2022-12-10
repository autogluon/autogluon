from abc import ABC
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from autogluon.common.features.types import *
from .base import AbstractVisualization
from .jupyter import JupyterMixin
from ..state import AnalysisState

__all__ = ["CorrelationVisualization", "CorrelationSignificanceVisualization", "FeatureInteractionVisualization"]


class _AbstractCorrelationChart(AbstractVisualization, JupyterMixin, ABC):
    def __init__(
        self,
        headers: bool = False,
        namespace: Optional[str] = None,
        fig_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(namespace, **kwargs)
        self.headers = headers
        if fig_args is None:
            fig_args = {}
        self.fig_args = fig_args

    def _render_internal(self, state: AnalysisState, render_key: str, header: str, chart_args: Dict[str, Any]) -> None:
        for ds, corr in state[render_key].items():
            # Don't render single cell
            if len(state.correlations[ds]) <= 1:
                continue

            if state.correlations_focus_field is not None:
                focus_field_header = f"; focus: absolute correlation for {state.correlations_focus_field} >= {state.correlations_focus_field_threshold}"
            else:
                focus_field_header = ""
            self.render_header_if_needed(state, f"{ds} - {state.correlations_method} {header}{focus_field_header}")

            fig, ax = plt.subplots(**self.fig_args)
            sns.heatmap(
                corr,
                annot=True,
                ax=ax,
                linewidths=0.9,
                linecolor="white",
                fmt=".2f",
                square=True,
                cbar_kws={"shrink": 0.5},
                **chart_args,
            )
            plt.yticks(rotation=0)
            plt.show(fig)


class CorrelationVisualization(_AbstractCorrelationChart):
    """
    Display feature correlations matrix.

    This report renders correlations between variable in a form of heatmap.
    The details of the report to be rendered depend on the configuration of
    :py:class:`~autogluon.eda.analysis.interaction.Correlation`

    Parameters
    ----------
    headers: bool, default = False
        if `True` then render headers
    namespace: str, default = None
        namespace to use; can be nested like `ns_a.ns_b.ns_c`
    fig_args: Optional[Dict[str, Any]] = None,
        kwargs to pass into chart figure

    See Also
    --------
    :py:class:`~autogluon.eda.analysis.interaction.Correlation`
    """

    def can_handle(self, state: AnalysisState) -> bool:
        return "correlations" in state

    def _render(self, state: AnalysisState) -> None:
        args = {"vmin": 0 if state.correlations_method == "phik" else -1, "vmax": 1, "center": 0, "cmap": "Spectral"}
        self._render_internal(state, "correlations", "correlation matrix", args)


class CorrelationSignificanceVisualization(_AbstractCorrelationChart):
    """
    Display feature correlations significance matrix.

    This report renders correlations significance matrix in a form of heatmap.
    The details of the report to be rendered depend on the configuration of
    :py:class:`~autogluon.eda.analysis.interaction.Correlation` and
    :py:class:`~autogluon.eda.analysis.interaction.CorrelationSignificance` analyses.

    Parameters
    ----------
    headers: bool, default = False
        if `True` then render headers
    namespace: str, default = None
        namespace to use; can be nested like `ns_a.ns_b.ns_c`
    fig_args: Optional[Dict[str, Any]] = None,
        kwargs to pass into chart figure

    See Also
    --------
    :py:class:`~autogluon.eda.analysis.interaction.Correlation`
    :py:class:`~autogluon.eda.analysis.interaction.CorrelationSignificance`
    """

    def can_handle(self, state: AnalysisState) -> bool:
        return "significance_matrix" in state

    def _render(self, state: AnalysisState) -> None:
        args = {"center": 3, "vmax": 5, "cmap": "Spectral", "robust": True}
        self._render_internal(state, "significance_matrix", "correlation significance matrix", args)


class FeatureInteractionVisualization(AbstractVisualization, JupyterMixin):
    def __init__(
        self,
        key: str,
        headers: bool = False,
        namespace: Optional[str] = None,
        numeric_as_categorical_threshold=20,
        fig_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(namespace, **kwargs)
        self.key = key
        self.headers = headers
        self.numeric_as_categorical_threshold = numeric_as_categorical_threshold
        if fig_args is None:
            fig_args = {}
        self.fig_args = fig_args

    def can_handle(self, state: AnalysisState) -> bool:
        return self.all_keys_must_be_present(state, "interactions", "raw_type")

    def __get(self, ds, df, state, interaction, param):
        value = interaction["features"].get(param, None)
        value_type = map_raw_type_to_feature_type(
            value, state.raw_type[ds].get(value, None), df, self.numeric_as_categorical_threshold
        )
        return value, value_type

    def _render(self, state: AnalysisState) -> None:
        for ds in state.interactions.keys():
            if self.key not in state.interactions[ds]:
                continue
            interaction = state.interactions[ds][self.key]
            df = interaction["data"].copy()
            x, x_type = self.__get(ds, df, state, interaction, "x")
            y, y_type = self.__get(ds, df, state, interaction, "y")
            hue, hue_type = self.__get(ds, df, state, interaction, "hue")

            # swap y <-> hue when category vs category to enable charting
            if (x_type is not None) and y_type == "category" and hue is None:
                hue, hue_type = y, y_type
                y, y_type = None, None

            chart_type = self._get_chart_type(x_type, y_type, hue_type)
            if chart_type is not None:
                chart_args = {
                    "x": x,
                    "y": y,
                    "hue": hue,
                    **self._kwargs.get(self.key, {}),
                }
                chart_args = {k: v for k, v in chart_args.items() if v is not None}
                if chart_type != "kdeplot":
                    chart_args.pop("fill", None)
                if chart_type == "barplot":
                    # Don't show ci ticks
                    chart_args["ci"] = None

                fig, ax = plt.subplots(**self.fig_args.get(self.key, {}))

                # convert to categoricals for plots
                for col, typ in zip([x, y, hue], [x_type, y_type, hue_type]):
                    if typ == "category":
                        df[col] = df[col].astype("object")

                data = df
                single_var = False
                if y is None and hue is None:
                    single_var = True
                    if x_type == "numeric":
                        data = df[x]
                        chart_args.pop("x")
                elif x is None and hue is None:
                    single_var = True
                    if y_type == "numeric":
                        data = df[y]
                        chart_args.pop("y")

                # Handling fitted distributions if present
                dists = None
                if (
                    (chart_type == "histplot")
                    and ("distributions_fit" in state)
                    and ((x_type, y_type, hue_type) == ("numeric", None, None))
                ):
                    chart_args["stat"] = "density"
                    dists = state.distributions_fit[ds][x]

                chart = self._get_sns_chart_method(chart_type)(ax=ax, data=data, **chart_args)
                if chart_type in ("countplot", "barplot", "boxplot"):
                    plt.setp(chart.get_xticklabels(), rotation=90)

                if dists is not None:
                    x_min, x_max = ax.get_xlim()
                    xs = np.linspace(x_min, x_max, 200)
                    for dist, v in dists.items():
                        _dist = getattr(stats, dist)
                        ax.plot(
                            xs,
                            _dist.pdf(xs, *dists[dist]["param"]),
                            ls="--",
                            lw=0.7,
                            label=f'{dist}: pvalue {dists[dist]["pvalue"]:.2f}',
                        )
                    ax.set_xlim(x_min, x_max)  # set the limits back to the ones of the distplot
                    plt.legend()

                if chart_type == "countplot":
                    for container in ax.containers:
                        ax.bar_label(container)

                if self.headers:
                    features = "/".join([interaction["features"][k] for k in ["x", "y", "hue"] if k in interaction["features"]])
                    prefix = "" if single_var else "Feature interaction between "
                    self.render_header_if_needed(state, f"{prefix}{features} in {ds}")
                plt.show(fig)

    def _get_sns_chart_method(self, chart_type):
        return {
            "countplot": sns.countplot,
            "barplot": sns.barplot,
            "boxplot": sns.boxplot,
            "kdeplot": sns.kdeplot,
            "scatterplot": sns.scatterplot,
            "regplot": sns.regplot,
            "histplot": sns.histplot,
        }.get(chart_type)

    def _get_chart_type(self, x_type: str, y_type: str, hue_type: Optional[str]) -> Optional[str]:
        types = {
            ("numeric", None, None): "histplot",
            ("category", None, None): "countplot",
            (None, "category", None): "countplot",
            ("category", None, "category"): "countplot",
            (None, "category", "category"): "countplot",
            ("numeric", None, "category"): "histplot",
            (None, "numeric", "category"): "histplot",
            ("category", "category", None): "barplot",
            ("category", "category", "category"): "barplot",
            ("category", "numeric", None): "boxplot",
            ("numeric", "category", None): "kdeplot",
            ("category", "numeric", "category"): "boxplot",
            ("numeric", "category", "category"): "kdeplot",
            ("numeric", "numeric", None): "regplot",
            ("numeric", "numeric", "category"): "scatterplot",
        }
        return types.get((x_type, y_type, hue_type), None)


def map_raw_type_to_feature_type(
    col: str, raw_type: str, series: pd.DataFrame, numeric_as_categorical_threshold: int = 20
) -> Optional[str]:
    if col is None:
        return None
    elif series[col].nunique() <= numeric_as_categorical_threshold:
        return "category"
    elif raw_type in [R_INT, R_FLOAT]:
        return "numeric"
    elif raw_type in [R_OBJECT, R_CATEGORY, R_BOOL]:
        return "category"
    else:
        return None
