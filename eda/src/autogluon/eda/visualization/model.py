from typing import Union, Dict, Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from autogluon.core.constants import REGRESSION
from .base import AbstractVisualization
from .jupyter import JupyterMixin
from ..state import AnalysisState

__all__ = ['ConfusionMatrix', 'FeatureImportance', 'RegressionEvaluation']


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


class RegressionEvaluation(AbstractVisualization, JupyterMixin):
    def __init__(self,
                 headers: bool = False,
                 fig_args: Union[None, Dict[str, Any]] = {},
                 chart_args: Union[None, Dict[str, Any]] = {},
                 namespace: str = None,
                 **kwargs) -> None:
        super().__init__(namespace, **kwargs)
        self.headers = headers
        self.fig_args = fig_args
        self.chart_args = chart_args

    def can_handle(self, state: AnalysisState) -> bool:
        return 'model_evaluation' in state and state.model_evaluation.problem_type == REGRESSION

    def _render(self, state: AnalysisState) -> None:
        data = pd.DataFrame({
            'y_true': state.model_evaluation.y_true,
            'y_pred': state.model_evaluation.y_pred
        })
        fig, ax = plt.subplots(**self.fig_args)
        sns.regplot(ax=ax, data=data, x='y_true', y='y_pred', **self.chart_args)
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
        self.show_barplots = show_barplots

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
