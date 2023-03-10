import logging
from typing import List, Optional

import pandas as pd
import shap
from fastshap import KernelExplainer

from autogluon.eda import AnalysisState
from autogluon.eda.analysis.base import AbstractAnalysis

__all__ = ["ShapAnalysis", "FastShapAnalysis"]

logger = logging.getLogger(__name__)


class _ShapAutogluonWrapper:
    def __init__(self, predictor, feature_names, target_class=None):
        self.ag_model = predictor
        self.feature_names = feature_names
        self.target_class = target_class
        if target_class is None and predictor.problem_type != "regression":
            logging.warning("Since target_class not specified, SHAP will explain predictions for each class")

    def predict_proba(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1, -1)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        if self.ag_model.problem_type == "regression":
            preds = self.ag_model.predict(X)
        else:
            preds = self.ag_model.predict_proba(X)
        if self.ag_model.problem_type == "regression" or self.target_class is None:
            return preds
        else:
            return preds[self.target_class]


class ShapAnalysis(AbstractAnalysis):
    def __init__(
        self,
        rows: pd.DataFrame,
        baseline_sample: int = 100,
        parent: Optional[AbstractAnalysis] = None,
        children: Optional[List[AbstractAnalysis]] = None,
        state: Optional[AnalysisState] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, children, state, **kwargs)
        self.rows = rows
        self.baseline_sample = baseline_sample

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return self.all_keys_must_be_present(args, "model", "train_data")

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        if self.baseline_sample <= len(args.train_data):
            _baseline_sample = self.baseline_sample
        else:
            _baseline_sample = len(args.train_data)

        baseline = args.train_data.sample(_baseline_sample, random_state=0)
        shap_data = []
        for _, row in self.rows.iterrows():
            _row = pd.DataFrame([row])
            if args.model.problem_type == "regression":
                predicted_class = 0
            else:
                predicted_class = args.model.predict(_row).iloc[0]
            ag_wrapper = _ShapAutogluonWrapper(args.model, args.train_data.columns, predicted_class)
            explainer = shap.KernelExplainer(ag_wrapper.predict_proba, baseline)
            ke_shap_values = explainer.shap_values(_row[args.train_data.columns], silent=True)
            shap_data.append(
                AnalysisState(
                    row=_row,
                    expected_value=explainer.expected_value,
                    shap_values=ke_shap_values[0],
                    features=row[args.train_data.columns],
                    feature_names=None,
                )
            )
        state.explain = {"shapley": shap_data}


class FastShapAnalysis(AbstractAnalysis):
    def __init__(
        self,
        rows: pd.DataFrame,
        baseline_sample: int = 100,
        parent: Optional[AbstractAnalysis] = None,
        children: Optional[List[AbstractAnalysis]] = None,
        state: Optional[AnalysisState] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, children, state, **kwargs)
        self.rows = rows
        assert baseline_sample >= 30
        self.baseline_sample = baseline_sample

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return self.all_keys_must_be_present(args, "model", "train_data")

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        if self.baseline_sample <= len(args.train_data):
            _baseline_sample = self.baseline_sample
        else:
            _baseline_sample = len(args.train_data)
        baseline = args.train_data.sample(_baseline_sample, random_state=0)
        data = baseline.drop(columns=args.model.label)

        def predict_fn(x):
            if args.model.problem_type == "regression":
                preds = args.model.predict(x, as_pandas=False)
            else:
                preds = args.model.predict_proba(x, as_pandas=False)
            return preds

        shap_data = []

        for _, row in self.rows.iterrows():
            _row = pd.DataFrame([row])
            ke = KernelExplainer(predict_fn, data)
            ke_shap_values = ke.calculate_shap_values(_row[data.columns], verbose=False)
            if args.model.problem_type == "regression":
                expected_value = ke_shap_values[0][-1]
                shap_values = ke_shap_values[0][:-1]
            else:
                predicted_class = ke_shap_values[0][-1].argmax()
                expected_value = ke_shap_values[0][-1][predicted_class]
                shap_values = ke_shap_values[0][:-1, predicted_class]

            features = _row[data.columns].to_numpy()[0]
            shap_data.append(
                AnalysisState(
                    row=_row,
                    expected_value=expected_value,
                    shap_values=shap_values,
                    features=features,
                    feature_names=list(data.columns),
                )
            )
        state.explain = {"shapley": shap_data}
