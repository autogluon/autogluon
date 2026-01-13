from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy as hc
from sklearn.inspection import PartialDependenceDisplay

from autogluon.common.features.types import R_BOOL, R_CATEGORY, R_DATETIME, R_FLOAT, R_INT, R_OBJECT

from ..state import AnalysisState
from .base import AbstractVisualization
from .jupyter import JupyterMixin

__all__ = [
    "CorrelationVisualization",
    "CorrelationSignificanceVisualization",
    "FeatureInteractionVisualization",
    "FeatureDistanceAnalysisVisualization",
    "PDPInteractions",
]


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
            cells_num = len(state.correlations[ds])
            if cells_num <= 1:
                continue

            fig_args = self.fig_args.copy()
            if "figsize" not in fig_args:
                fig_args["figsize"] = (cells_num, cells_num)

            if state.correlations_focus_field is not None:
                focus_field_header = f"; focus: absolute correlation for `{state.correlations_focus_field}` >= `{state.correlations_focus_field_threshold}`"
            else:
                focus_field_header = ""
            self.render_header_if_needed(
                state, f"`{ds}` - `{state.correlations_method}` {header}{focus_field_header}", ds=ds
            )

            fig, ax = plt.subplots(**fig_args)
            sns.heatmap(
                corr,
                annot=True,
                ax=ax,
                linewidths=0.5,
                linecolor="lightgrey",
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
    namespace: Optional[str], default = None
        namespace to use; can be nested like `ns_a.ns_b.ns_c`
    fig_args: Optional[Dict[str, Any]], default = None,
        kwargs to pass into chart figure

    See Also
    --------
    :py:class:`~autogluon.eda.analysis.interaction.Correlation`
    """

    def can_handle(self, state: AnalysisState) -> bool:
        return "correlations" in state

    def _render(self, state: AnalysisState) -> None:
        args = {
            **{"vmin": 0 if state.correlations_method == "phik" else -1, "vmax": 1, "center": 0, "cmap": "Spectral"},
            **self._kwargs,
        }
        self._render_internal(state, "correlations", "correlation matrix", args)


class _AbstractFeatureInteractionPlotRenderer(ABC):
    @abstractmethod
    def _render(self, state, ds, params, param_types, ax, data, chart_args):
        raise NotImplementedError  # pragma: no cover

    def render(self, state, ds, params, param_types, data, fig_args, chart_args):
        fig, ax = plt.subplots(**fig_args)
        self._render(state, ds, params, param_types, ax, data, chart_args)
        plt.show(fig)


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
    namespace: Optional[str], default = None
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


class FeatureDistanceAnalysisVisualization(AbstractVisualization, JupyterMixin):
    """
    Feature distance visualization.

    This component renders graphical representations of distances between features to highlight features that can be
    either simplified or completely removed.

    Parameters
    ----------
    headers: bool, default = False
        if `True` then render headers
    namespace: Optional[str], default = None
        namespace to use; can be nested like `ns_a.ns_b.ns_c`
    fig_args: Optional[Dict[str, Any]] = None,
        kwargs to pass into chart figure
    kwargs
    """

    def __init__(
        self,
        namespace: Optional[str] = None,
        fig_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(namespace, **kwargs)
        if fig_args is None:
            fig_args = {}
        self.fig_args = fig_args

    def can_handle(self, state: AnalysisState) -> bool:
        return self.all_keys_must_be_present(state, "feature_distance")

    def _render(self, state: AnalysisState) -> None:
        fig_args = self.fig_args.copy()
        if "figsize" not in fig_args:
            fig_args["figsize"] = (12, len(state.feature_distance.columns) / 4)

        fig, ax = plt.subplots(**fig_args)
        default_args = dict(orientation="left")
        ax.grid(False)
        hc.dendrogram(
            ax=ax,
            Z=state.feature_distance.linkage,
            labels=state.feature_distance.columns,
            leaf_font_size=10,
            **{**default_args, **self._kwargs},
        )
        plt.show(fig)
        if len(state.feature_distance.near_duplicates) > 0:
            message = (
                f"**The following feature groups are considered as near-duplicates**:\n\n"
                f"Distance threshold: <= `{state.feature_distance.near_duplicates_threshold}`. "
                f"Consider keeping only some of the columns within each group:\n"
            )
            for group in state.feature_distance.near_duplicates:
                message += f"\n - `{'`, `'.join(sorted(group['nodes']))}` - distance `{group['distance']:.2f}`"
            self.render_markdown(message)


class FeatureInteractionVisualization(AbstractVisualization, JupyterMixin):
    """
    Feature interaction visualization.

    This report renders feature interaction analysis results.
    The details of the report to be rendered depend on the variable types combination in `x`/`y`/`hue`.
    `key` is used to link analysis and visualization - this allows to have multiple analyses/visualizations in one composite analysis.

    Parameters
    ----------
    key: str
        key used to store the analysis in the state; the value is placed in the state by FeatureInteraction.
        If the key is not provided, then use one of theform: 'x:A|y:B|hue:C' (omit corresponding x/y/hue if the value not provided)
        See also :class:`autogluon.eda.analysis.interaction.FeatureInteraction`
    numeric_as_categorical_threshold
    headers: bool, default = False
        if `True` then render headers
    namespace: Optional[str], default = None
        namespace to use; can be nested like `ns_a.ns_b.ns_c`
    fig_args: Optional[Dict[str, Any]] = None,
        kwargs to pass into chart figure
    kwargs
        parameters to pass as a chart args

    See Also
    --------
    :py:class:`~autogluon.eda.analysis.interaction.FeatureInteraction`
    """

    def __init__(
        self,
        key: str,
        numeric_as_categorical_threshold: int = 20,
        max_categories_to_consider_render: int = 30,
        headers: bool = False,
        namespace: Optional[str] = None,
        fig_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(namespace, **kwargs)
        self.key = key
        self.headers = headers
        self.numeric_as_categorical_threshold = numeric_as_categorical_threshold
        self.max_categories_to_consider_render = max_categories_to_consider_render
        if fig_args is None:
            fig_args = {}
        self.fig_args = fig_args

    def can_handle(self, state: AnalysisState) -> bool:
        return self.all_keys_must_be_present(state, "interactions", "raw_type")

    def _render(self, state: AnalysisState) -> None:
        for ds in state.interactions.keys():
            if self.key not in state.interactions[ds]:
                continue
            interaction = state.interactions[ds][self.key]
            interaction_features = interaction["features"]
            df = interaction["data"].copy()
            x, x_type = self._get_value_and_type(ds, df, state, interaction_features, "x")
            y, y_type = self._get_value_and_type(ds, df, state, interaction_features, "y")
            hue, hue_type = self._get_value_and_type(ds, df, state, interaction_features, "hue")

            # Don't render high-cardinality category variables
            features = "/".join(
                [f"`{interaction_features[k]}`" for k in ["x", "y", "hue"] if k in interaction_features]
            )
            for f, t in [(x, x_type), (y, y_type), (hue, hue_type)]:
                if t == "category" and df[f].nunique() > self.max_categories_to_consider_render:
                    self.render_markdown(
                        f"Interaction {features} is not rendered due to `{f}` "
                        f"having too many categories (`{df[f].nunique()}` > `{self.max_categories_to_consider_render}`) "
                        f"to place on plot axis."
                    )
                    return

            y, y_type, hue, hue_type = self._swap_y_and_hue_if_necessary(x_type, y, y_type, hue, hue_type)

            renderer_cls: Optional[Type[_AbstractFeatureInteractionPlotRenderer]] = self._get_chart_renderer(
                x_type, y_type, hue_type
            )
            if renderer_cls is None:
                return
            renderer: _AbstractFeatureInteractionPlotRenderer = renderer_cls()  # Create instance

            df = self._convert_categoricals_to_objects(df, x, x_type, y, y_type, hue, hue_type)
            chart_args, data, is_single_var = self._prepare_chart_args(df, x, x_type, y, y_type, hue)

            if self.headers:
                prefix = "" if is_single_var else "Feature interaction between "
                self.render_header_if_needed(state, f"{prefix}{features} in `{ds}`", ds=ds)

            fig_args = self.fig_args.copy()
            if "figsize" not in fig_args:
                fig_args["figsize"] = (12, 6)

            renderer.render(
                state=state,
                ds=ds,
                params=(x, y, hue),
                param_types=(x_type, y_type, hue_type),
                data=data,
                fig_args=fig_args,
                chart_args=chart_args,
            )

    def _prepare_chart_args(self, df, x, x_type, y, y_type, hue) -> Tuple[Dict[str, Any], pd.DataFrame, bool]:
        chart_args = {"x": x, "y": y, "hue": hue, **self._kwargs}
        chart_args = {k: v for k, v in chart_args.items() if v is not None}
        data = df
        is_single_var = False
        if x is not None and y is None and hue is None:
            is_single_var = True
            if x_type == "numeric":
                data = df[x]
                chart_args.pop("x")
        elif y is not None and x is None and hue is None:
            is_single_var = True
            if y_type == "numeric":
                data = df[y]
                chart_args.pop("y")
        return chart_args, data, is_single_var

    def _convert_categoricals_to_objects(self, df, x, x_type, y, y_type, hue, hue_type):
        # convert to categoricals for plots
        for col, typ in zip([x, y, hue], [x_type, y_type, hue_type]):
            if typ == "category":
                df[col] = df[col].astype("object")
        return df

    def _swap_y_and_hue_if_necessary(self, x_type, y, y_type, hue, hue_type):
        # swap y <-> hue when category vs category is provided and no hue is specified
        if (x_type is not None) and y_type == "category" and hue_type is None:
            hue, hue_type = y, y_type
            y, y_type = None, None
        return y, y_type, hue, hue_type

    def _get_value_and_type(
        self, ds: str, df: pd.DataFrame, state: AnalysisState, interaction_features: Dict[str, Any], param: str
    ) -> Tuple[Any, Optional[str]]:
        col = interaction_features.get(param, None)
        value_type = self._map_raw_type_to_feature_type(
            col, state.raw_type[ds].get(col, None), df, self.numeric_as_categorical_threshold
        )
        return col, value_type

    def _get_chart_renderer(
        self, x_type: Optional[str], y_type: Optional[str], hue_type: Optional[str]
    ) -> Optional[Type[_AbstractFeatureInteractionPlotRenderer]]:
        types = {
            ("numeric", None, None): self._HistPlotRenderer,
            ("category", None, None): self._CountPlotRenderer,
            (None, "category", None): self._CountPlotRenderer,
            ("category", None, "category"): self._CountPlotRenderer,
            (None, "category", "category"): self._CountPlotRenderer,
            ("numeric", None, "category"): self._HistPlotRenderer,
            (None, "numeric", "category"): self._HistPlotRenderer,
            ("category", "category", None): self._BarPlotRenderer,
            ("category", "category", "category"): self._BarPlotRenderer,
            ("category", "numeric", None): self._BoxPlotRenderer,
            ("numeric", "category", None): self._KdePlotRenderer,
            ("category", "numeric", "category"): self._BoxPlotRenderer,
            ("numeric", "category", "category"): self._KdePlotRenderer,
            ("numeric", "numeric", None): self._RegPlotRenderer,
            ("numeric", "numeric", "category"): self._ScatterPlotRenderer,
            ("numeric", "numeric", "numeric"): self._ScatterPlotRenderer,
            ("datetime", "numeric", None): self._LinePlotRenderer,
        }
        return types.get((x_type, y_type, hue_type), None)

    def _map_raw_type_to_feature_type(
        self, col: str, raw_type: str, df: pd.DataFrame, numeric_as_categorical_threshold: int = 20
    ) -> Optional[str]:
        if col is None:
            return None
        elif df[col].nunique() <= numeric_as_categorical_threshold:
            return "category"
        elif raw_type in [R_INT, R_FLOAT]:
            return "numeric"
        elif raw_type in [R_DATETIME]:
            return "datetime"
        elif raw_type in [R_OBJECT, R_CATEGORY, R_BOOL]:
            return "category"
        else:
            return None

    class _HistPlotRenderer(_AbstractFeatureInteractionPlotRenderer):
        def _render(self, state, ds, params, param_types, ax, data, chart_args, num_point_to_fit=200):
            x = params[0]
            fitted_distributions_present = (
                ("distributions_fit" in state)
                and (param_types == ("numeric", None, None))  # (x, y, hue)
                and (state.distributions_fit[ds].get(x, None) is not None)
            )

            if "stat" not in chart_args:
                chart_args["stat"] = "density"
            sns.histplot(ax=ax, data=data, **chart_args)

            if fitted_distributions_present:  # types for  x, y, hue
                dists = state.distributions_fit[ds][x]
                x_min, x_max = ax.get_xlim()
                xs = np.linspace(x_min, x_max, num_point_to_fit)
                for dist, v in dists.items():
                    _dist = getattr(stats, dist)
                    ax.plot(
                        xs,
                        _dist.pdf(xs, *v["param"]),
                        ls="--",
                        label=f"{dist}: pvalue {v['pvalue']:.2f}",
                    )
                ax.set_xlim(x_min, x_max)  # set the limits back to the ones of the distplot
                plt.legend()

    class _KdePlotRenderer(_AbstractFeatureInteractionPlotRenderer):
        def _render(self, state, ds, params, param_types, ax, data, chart_args):
            chart_args.pop("fill", None)
            chart = sns.kdeplot(ax=ax, data=data, **chart_args)
            plt.setp(chart.get_xticklabels(), rotation=90)

    class _BoxPlotRenderer(_AbstractFeatureInteractionPlotRenderer):
        def _render(self, state, ds, params, param_types, ax, data, chart_args):
            chart = sns.boxplot(ax=ax, data=data, **chart_args)
            plt.setp(chart.get_xticklabels(), rotation=90)

    class _CountPlotRenderer(_AbstractFeatureInteractionPlotRenderer):
        def _render(self, state, ds, params, param_types, ax, data, chart_args):
            chart = sns.countplot(ax=ax, data=data, **chart_args)
            plt.setp(chart.get_xticklabels(), rotation=90)
            for container in ax.containers:
                ax.bar_label(container)

    class _BarPlotRenderer(_AbstractFeatureInteractionPlotRenderer):
        def _render(self, state, ds, params, param_types, ax, data, chart_args):
            chart_args["errorbar"] = None  # Don't show ci ticks
            chart = sns.barplot(ax=ax, data=data, **chart_args)
            plt.setp(chart.get_xticklabels(), rotation=90)

    class _ScatterPlotRenderer(_AbstractFeatureInteractionPlotRenderer):
        def _render(self, state, ds, params, param_types, ax, data, chart_args):
            sns.scatterplot(ax=ax, data=data, **chart_args)

    class _RegPlotRenderer(_AbstractFeatureInteractionPlotRenderer):
        def _render(self, state, ds, params, param_types, ax, data, chart_args):
            sns.regplot(ax=ax, data=data, **chart_args)

    class _LinePlotRenderer(_AbstractFeatureInteractionPlotRenderer):
        def _render(self, state, ds, params, param_types, ax, data, chart_args):
            sns.lineplot(ax=ax, data=data, **chart_args)


class PDPInteractions(AbstractVisualization, JupyterMixin):
    """
    Display Partial Dependence Plots (PDP) with Individual Conditional Expectation (ICE)

    The visualizations have two modes:
    - regular PDP + ICE plots - this is the default mode of operation
    - two-way PDP plots - this mode can be selected via passing two `features` and setting `two_way = True`

    ICE plots complement PDP by showing the relationship between a feature and the model's output for each individual instance in the dataset.
    ICE lines (blue) can be overlaid on PDPs (red) to provide a more detailed view of how the model behaves for specific instances.

    Parameters
    ----------
    features: Union[str, List[str]]
        feature to display on the plots
    two_way: bool, default = False
        render two-way PDP; this mode works only when two `features` are specified
    target: Optional[Any], default = None
        In a multiclass setting, specifies the class for which the PDPs should be computed.
        Ignored in binary classification or classical regression settings
    namespace: Optional[str], default = None
        namespace to use; can be nested like `ns_a.ns_b.ns_c`
    fig_args: Optional[Dict[str, Any]] = None,
        kwargs to pass into chart figure
    headers: bool, default = False
        if `True` then render headers
    sample: Union[None, int, float], default = None
        sample size; if `int`, then row number is used;
        `float` must be between 0.0 and 1.0 and represents fraction of dataset to sample;
        `None` means no sampling
        See also :func:`autogluon.eda.analysis.dataset.Sampler`
    kwargs
    """

    MAX_CHARTS_PER_ROW = 2

    def __init__(
        self,
        features: Union[str, List[str]],
        two_way: bool = False,
        target: Optional[Any] = None,
        namespace: Optional[str] = None,
        fig_args: Optional[Dict[str, Dict[str, Any]]] = None,
        sample: Optional[Union[float, int]] = 300,
        headers: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(namespace, **kwargs)

        if type(features) is not list:
            features = [features]  # type: ignore
        self.features: list = features

        self.target = target
        self.fig_args = fig_args
        self.sample = sample
        self.headers = headers
        self.two_way = two_way

        if two_way:
            assert len(self.features) == 2, "`two_way` can only be used if only 2 `features` are passed"

    def can_handle(self, state: AnalysisState) -> bool:
        return self.all_keys_must_be_present(state, "model", "pdp_data")

    def _render(self, state: AnalysisState) -> None:
        additional_kwargs, fig_args, features = self._get_args()

        if self.headers:
            self.render_header_if_needed(state, "Partial Dependence Plots")

        fig, axs = plt.subplots(**fig_args)

        kwargs = {**additional_kwargs, **self._kwargs}
        data = state.pdp_data

        if self.two_way:
            for f in self.features[:-1]:
                data = data[data[f].notna()]

        PartialDependenceDisplay.from_estimator(
            _SklearnAutoGluonWrapper(state.model),
            data,
            self.features,
            ax=axs.ravel()[: len(features)],
            target=self.target,
            subsample=self.sample,
            **kwargs,
        )
        plt.tight_layout(h_pad=0.3, w_pad=0.5)
        plt.show()

    def _get_args(self):
        n = len(self.features)
        cols = self.MAX_CHARTS_PER_ROW if n > self.MAX_CHARTS_PER_ROW else n
        rows = int(np.ceil(n / cols))
        if self.fig_args is None:
            self.fig_args = {}
        kind = "both"
        additional_kwargs: Dict[str, Any] = {}
        features = self.features

        if self.two_way:
            fig_args = {**dict(nrows=1, ncols=3, figsize=(12, 3)), **self.fig_args}
            if len(self.features) == 2:
                kind = "average"
            features.append(features.copy())
        else:
            fig_args = {**dict(nrows=rows, ncols=cols, figsize=(12, 3 * rows)), **self.fig_args}
            additional_kwargs["pd_line_kw"] = {"color": "red"}
            additional_kwargs["ice_lines_kw"] = {"color": "blue"}
        additional_kwargs["kind"] = kind

        return additional_kwargs, fig_args, features


class _SklearnAutoGluonWrapper:
    def __init__(self, estimator):
        self.estimator = estimator
        self.estimator_type = "regressor" if estimator.problem_type == "regression" else "classifier"

    @property
    def _estimator_type(self):
        return self.estimator_type

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, X, Y=None):
        self.estimator.fit(X)

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    @property
    def classes_(self):
        return self.estimator.class_labels
