from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import seaborn as sns

from .. import AnalysisState
from .base import AbstractVisualization
from .jupyter import JupyterMixin

__all__ = ["AnomalyScoresVisualization"]

from ..utils.common import expand_nested_args_into_nested_maps, get_empty_dict_if_none


class AnomalyScoresVisualization(AbstractVisualization, JupyterMixin):
    """
    Visualize anomaly scores across datasets.

    The report depends on :py:class:`~autogluon.eda.analysis.anomaly.AnomalyDetectorAnalysis`,

    Parameters
    ----------
    threshold_stds: float = 3,
        defines how many standard deviations from mean the scores will be marked as anomalies
    headers: bool, default = False
        if `True` then render headers
    namespace: Optional[str], default = None
        namespace to use; can be nested like `ns_a.ns_b.ns_c`
    fig_args: Optional[Dict[str, Any]] = None,
        kwargs to pass into visualization component.
    chart_args
        kwargs to pass into visualization component
        The chart contains two scatterpolots combined: normal and anomaly data points; both can be customized via passing
        additional arguments in `chart_args` - see the example below.


    Examples
    --------
    >>> import autogluon.eda.analysis as eda
    >>> import autogluon.eda.visualization as viz
    >>> import autogluon.eda.auto as auto
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> df_train = pd.DataFrame(...)
    >>> df_test = pd.DataFrame(...)
    >>> label = 'target'
    >>> threshold_stds = 3  # mark 3 standard deviations score values as anomalies
    >>>
    >>> chart_args={
    >>>     'normal.color': 'lightgrey',
    >>>     'anomaly.color': 'orange',
    >>> }
    >>>
    >>> state = auto.analyze(
    >>>     train_data=df_train,
    >>>     test_data=df_test,
    >>>     label=label,
    >>>     return_state=True,
    >>>     anlz_facets=[
    >>>         eda.dataset.ProblemTypeControl(),
    >>>         eda.transform.ApplyFeatureGenerator(category_to_numbers=True, children=[
    >>>             eda.anomaly.AnomalyDetectorAnalysis(
    >>>                 store_explainability_data=True  # Store additional functions for explainability
    >>>             ),
    >>>         ])
    >>>     ],
    >>>     viz_facets=[
    >>>         viz.anomaly.AnomalyScoresVisualization(
    >>>             threshold_stds=threshold_stds,
    >>>             headers=True,
    >>>             fig_args=dict(figsize=(8, 4)),
    >>>             **chart_args, # pass chart args customizations
    >>>         )
    >>>     ]
    >>> )
    >>>
    >>> # explain top anomalies
    >>> train_anomaly_scores = state.anomaly_detection.scores.train_data
    >>> anomaly_idx = train_anomaly_scores[train_anomaly_scores >= train_anomaly_scores.std() * threshold_stds]
    >>> anomaly_idx = anomaly_idx.sort_values(ascending=False).index
    >>>
    >>> auto.explain_rows(
    >>>     # Use helper function stored via `store_explainability_data=True`
    >>>     **state.anomaly_detection.explain_rows_fns.train_data(anomaly_idx[:3]),
    >>>     plot='waterfall',
    >>> )

    """

    def __init__(
        self,
        threshold_stds: float = 3,
        headers: bool = False,
        namespace: Optional[str] = None,
        fig_args: Optional[Dict[str, Any]] = None,
        **chart_args,
    ) -> None:
        super().__init__(namespace, **chart_args)
        self.threshold_stds = threshold_stds
        self.headers = headers
        if fig_args is None:
            fig_args = {}
        self.fig_args = fig_args
        self.chart_args = expand_nested_args_into_nested_maps(get_empty_dict_if_none(chart_args))

    def can_handle(self, state: AnalysisState) -> bool:
        return self.all_keys_must_be_present(state, "anomaly_detection")

    def _render(self, state: AnalysisState) -> None:
        scores = state.anomaly_detection.scores
        threshold = scores.train_data.std() * self.threshold_stds
        for ds, ds_scores in scores.items():
            self.render_header_if_needed(
                state, f"`{ds}` anomalies for {self.threshold_stds}-sigma outlier scores", ds=ds
            )
            data = ds_scores.reset_index(drop=True).reset_index()

            fig, ax = plt.subplots(**self.fig_args)

            common_chart_args = dict(ax=ax, x="index", y="score")

            sns.scatterplot(
                data=data[data.score < threshold], **common_chart_args, **self.chart_args.get("normal", {})
            )
            sns.scatterplot(
                data=data[data.score >= threshold], **common_chart_args, **self.chart_args.get("anomaly", {})
            )

            ax.axhline(
                y=threshold,
                color="r",
                linestyle="--",
            )
            ax.text(
                x=0,
                y=threshold,
                s=f"{threshold:.4f}",
                color="red",
                rotation="vertical",
                horizontalalignment="right",
                verticalalignment="top",
            )
            plt.tight_layout(h_pad=0.3, w_pad=0.5)
            plt.show(fig)
