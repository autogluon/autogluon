from typing import Any, Dict, Optional

import matplotlib.pyplot as plt

from .. import AnalysisState
from .base import AbstractVisualization
from .jupyter import JupyterMixin


class AnomalyVisualization(AbstractVisualization, JupyterMixin):
    def __init__(
        self,
        threshold_stds: float = 3,
        headers: bool = False,
        namespace: str = None,
        fig_args: Optional[Dict[str, Any]] = None,
        **chart_args,
    ) -> None:
        super().__init__(namespace, **chart_args)
        self.threshold_stds = threshold_stds
        self.headers = headers
        if fig_args is None:
            fig_args = {}
        self.fig_args = fig_args
        self.chart_args = chart_args

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

            chart_args = {**dict(s=5), **self.chart_args, **dict(ax=ax, kind="scatter", x="index", y="score")}
            ax = data[data.score < threshold].plot(**chart_args)
            data[data.score >= threshold].plot(**chart_args, c="orange")
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
