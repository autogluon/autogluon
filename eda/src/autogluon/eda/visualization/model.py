from typing import Union, Dict, Any

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from autogluon.eda import AnalysisState
from autogluon.eda.visualization import AbstractVisualization, JupyterMixin


class ConfusionMatrix(AbstractVisualization, JupyterMixin):
    def __init__(self,
                 headers: bool = False,
                 fig_args: Union[None, Dict[str, Any]] = {},
                 namespace: str = None,
                 **kwargs) -> None:
        super().__init__(namespace, **kwargs)
        self.headers = headers
        self.fig_args = fig_args

    def can_handle(self, state: AnalysisState) -> bool:
        return 'model_evaluation' in state and 'confusion_matrix' in state.model_evaluation

    def _render(self, state: AnalysisState) -> None:
        self.render_header_if_needed(state, 'Confusion matrix')
        cm = pd.DataFrame(state.model_evaluation.confusion_matrix)
        cm.index.name = 'Actual'
        cm.columns.name = 'Predicted'
        normalized = state.model_evaluation.confusion_matrix_normalized
        fmt = ',.2%' if normalized else 'd'
        fig, ax = plt.subplots(**self.fig_args)
        sns.heatmap(cm, ax=ax, cmap="Blues", annot=True, fmt=fmt, cbar=False, **self._kwargs)
        plt.show(fig)


class FeatureImportance(AbstractVisualization, JupyterMixin):
    def __init__(self,
                 headers: bool = False,
                 fig_args: Union[None, Dict[str, Any]] = {},
                 show_barplots: bool = False,
                 namespace: str = None,
                 **kwargs) -> None:
        super().__init__(namespace, **kwargs)
        self.headers = headers
        self.fig_args = fig_args
        self.show_barplots=show_barplots

    def can_handle(self, state: AnalysisState) -> bool:
        return 'model_evaluation' in state and 'importance' in state.model_evaluation

    def _render(self, state: AnalysisState) -> None:
        self.render_header_if_needed(state, 'Feature importance')
        importance = state.model_evaluation.importance
        self.display_obj(importance)
        if self.show_barplots:
            fig, ax = plt.subplots(**self.fig_args)
            sns.barplot(ax=ax, data=importance.reset_index(), y='index', x='importance')
            plt.show(fig)

