import logging
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import shap

from autogluon.core.constants import REGRESSION
from autogluon.eda import AnalysisState
from autogluon.eda.analysis.base import AbstractAnalysis

__all__ = ["ShapAnalysis"]

logger = logging.getLogger(__name__)


class _ShapAutoGluonWrapper:
    def __init__(self, predictor, feature_names, target_class=None):
        self.ag_model = predictor
        self.feature_names = feature_names
        self.target_class = target_class
        if target_class is None and predictor.problem_type != REGRESSION:
            logging.warning("Since target_class not specified, SHAP will explain predictions for each class")

    def predict_proba(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1, -1)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        if self.ag_model.problem_type == REGRESSION:
            preds = self.ag_model.predict(X)
        else:
            preds = self.ag_model.predict_proba(X)
        if self.ag_model.problem_type == REGRESSION or self.target_class is None:
            return preds
        else:
            return preds[self.target_class]


class ShapAnalysis(AbstractAnalysis):
    """
    Perform Shapley values calculation using `shap` package for the given rows.

    Parameters
    ----------
    rows: pd.DataFrame,
        rows to explain
    baseline_sample: int, default = 100
        The background dataset size to use for integrating out features. To determine the impact
        of a feature, that feature is set to "missing" and the change in the model output
        is observed.
    parent: Optional[AbstractAnalysis], default = None
        parent Analysis
    children: List[AbstractAnalysis], default = []
        wrapped analyses; these will receive sampled `args` during `fit` call
    state: AnalysisState
        state to be updated by this fit function
    random_state: int, default = 0
        random state for sampling
    kwargs

    Examples
    --------
    >>> import autogluon.eda.analysis as eda
    >>> import autogluon.eda.visualization as viz
    >>> import autogluon.eda.auto as auto
    >>>
    >>> auto.analyze(
    >>>     train_data=..., model=...,
    >>>     anlz_facets=[
    >>>         eda.explain.ShapAnalysis(rows, baseline_sample=200),
    >>>     ],
    >>>     viz_facets=[
    >>>         # Visualize the given SHAP values with an additive force layout
    >>>         viz.explain.ExplainForcePlot(),
    >>>         # Visualize the given SHAP values with a waterfall layout
    >>>         viz.explain.ExplainWaterfallPlot(),
    >>>     ]
    >>> )

    See Also
    --------
    :py:class:`~shap.KernelExplainer`
    :py:class:`~autogluon.eda.visualization.explain.ExplainForcePlot`
    :py:class:`~autogluon.eda.visualization.explain.ExplainWaterfallPlot`
    """

    def __init__(
        self,
        rows: pd.DataFrame,
        baseline_sample: int = 100,
        parent: Optional[AbstractAnalysis] = None,
        children: Optional[List[AbstractAnalysis]] = None,
        state: Optional[AnalysisState] = None,
        random_state: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(parent, children, state, **kwargs)
        self.rows = rows
        self.baseline_sample = baseline_sample
        self.random_state = random_state

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return self.all_keys_must_be_present(args, "model", "train_data")

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        if self.baseline_sample <= len(args.train_data):
            _baseline_sample = self.baseline_sample
        else:
            _baseline_sample = len(args.train_data)

        baseline = args.train_data.sample(_baseline_sample, random_state=self.random_state)
        shap_data = []
        for _, row in self.rows.iterrows():
            _row = pd.DataFrame([row])
            if args.model.problem_type == REGRESSION:
                predicted_class = None
            else:
                predicted_class = args.model.predict(_row).iloc[0]
            ag_wrapper = _ShapAutoGluonWrapper(args.model, args.train_data.columns, predicted_class)
            explainer = shap.KernelExplainer(ag_wrapper.predict_proba, baseline)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                # Suppress sklearn pipeline warnings
                np.int = int  # type: ignore[attr-defined] # workaround to address shap's use of old numpy APIs
                ke_shap_values = explainer.shap_values(_row[args.train_data.columns], silent=True)
            shap_data.append(
                AnalysisState(
                    row=_row,
                    expected_value=explainer.expected_value,
                    shap_values=ke_shap_values[0],
                    features=row[args.model.original_features],
                    feature_names=None,
                )
            )
        state.explain = {"shapley": shap_data}
