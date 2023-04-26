from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from autogluon.core.constants import (
    BINARY,
    MULTICLASS,
    PROBLEM_TYPES_CLASSIFICATION,
    PROBLEM_TYPES_REGRESSION,
    REGRESSION,
)
from autogluon.tabular import TabularPredictor

from .base import AbstractAnalysis, AnalysisState

__all__ = ["AutoGluonModelEvaluator", "AutoGluonModelQuickFit"]


class AutoGluonModelQuickFit(AbstractAnalysis):
    """
    Fit a quick model using AutoGluon.

    `train_data`, `val_data` and `label` must be present in args.

    Note: this component can be wrapped into :py:class:`~autogluon.eda.analysis.dataset.TrainValidationSplit` and `~autogluon.eda.analysis.dataset.Sampler`
    to perform automated sampling and train-test split. This whole logic is implemented in :py:meth:`~autogluon.eda.auto.simple.quick_fit` shortcut.

    Examples
    --------
    >>> import autogluon.eda.analysis as eda
    >>>
    >>> # Quick fit
    >>> state = auto.quick_fit(
    >>>     train_data=..., label=...,
    >>>     return_state=True,  # return state object from call
    >>>     hyperparameters={'GBM': {}}  # train specific model
    >>> )
    >>>
    >>> # Using quick fit model
    >>> model = state.model
    >>> y_pred = model.predict(test_data)

    Parameters
    ----------
    problem_type: str, default = 'auto'
        problem type to use. Valid problem_type values include ['auto', 'binary', 'multiclass', 'regression', 'quantile', 'softclass']
        auto means it will be Auto-detected using AutoGluon methods.
    estimator_args: Optional[Dict[str, Any]], default = None,
        kwargs to pass into estimator constructor (`TabularPredictor`)
    save_model_to_state: bool, default = True,
        save fitted model into `state` under `model` key.
        This functionality might be helpful in cases when the fitted model could be usable for other purposes (i.e. imputers)
    parent: Optional[AbstractAnalysis], default = None
        parent Analysis
    children: Optional[List[AbstractAnalysis]], default None
        wrapped analyses; these will receive sampled `args` during `fit` call
    kwargs

    See Also
    --------
    :py:meth:`~autogluon.eda.auto.simple.quick_fit`
    :py:class:`~autogluon.eda.analysis.dataset.TrainValidationSplit`
    :py:class:`~autogluon.eda.analysis.dataset.Sampler`

    """

    def __init__(
        self,
        problem_type: str = "auto",
        estimator_args: Optional[Dict[str, Any]] = None,
        parent: Optional[AbstractAnalysis] = None,
        children: Optional[List[AbstractAnalysis]] = None,
        save_model_to_state: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(parent, children, **kwargs)

        valid_problem_types = ["auto"] + PROBLEM_TYPES_REGRESSION + PROBLEM_TYPES_CLASSIFICATION
        assert problem_type in valid_problem_types, f"Valid problem_type values include {valid_problem_types}"
        self.problem_type: Optional[str] = None if problem_type == "auto" else problem_type

        self.save_model_to_state = save_model_to_state

        if estimator_args is not None:
            self.estimator_args = estimator_args
        else:
            self.estimator_args = {}

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return self.all_keys_must_be_present(args, "train_data", "val_data", "label")

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        estimator: TabularPredictor = TabularPredictor(
            label=args.label, problem_type=self.problem_type, **self.estimator_args
        )
        estimator.fit(train_data=args.train_data, **self.args)
        self.args["model"] = estimator

        if self.save_model_to_state:
            state["model"] = estimator


class AutoGluonModelEvaluator(AbstractAnalysis):
    """
    Evaluate AutoGluon model performance.

    This analysis requires a trained classifier passed in `model` arg and uses 'val_data' dataset to assess model performance.

    It is assumed that the validation dataset should follow the same column names seen by the model and has not been used during the training process.

    Parameters
    ----------
    model: TabularPredictor, required
        fitted AutoGluon model to analyze
    val_data: pd.DataFrame, required
        validation dataset to use.
        Warning: do not use data used for training as a validation data.
        Predictions on data used by the model during training tend to be optimistic and might not generalize on unseen data.
    normalize : {'true', 'pred', 'all'}, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized.
        Note: applicable only for binary and multiclass classification; ignored for regression models.
    parent: Optional[AbstractAnalysis], default = None
        parent Analysis
    children: Optional[List[AbstractAnalysis]], default None
        wrapped analyses; these will receive sampled `args` during `fit` call

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
    >>>     viz.layouts.MarkdownSectionComponent(markdown=f'### Model Prediction for {predictor.label}'),
    >>>     viz.model.ConfusionMatrix(fig_args=dict(figsize=(3,3)), annot_kws={"size": 12}),
    >>>     viz.model.RegressionEvaluation(fig_args=dict(figsize=(6,6)), chart_args=dict(marker='o', scatter_kws={'s':5})),
    >>>     viz.layouts.MarkdownSectionComponent(markdown=f'### Feature Importance for Trained Model'),
    >>>     viz.model.FeatureImportance(show_barplots=True)
    >>> ])
    """

    def __init__(
        self,
        normalize: Union[None, str] = None,
        parent: Optional[AbstractAnalysis] = None,
        children: Optional[List[AbstractAnalysis]] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, children, **kwargs)
        self.normalize = normalize

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        keys_present = self.all_keys_must_be_present(args, "model", "val_data")
        data_cols = sorted(args.val_data.columns.values)
        model_cols = sorted(args.model.original_features + [args.model.label])
        columns_the_same = data_cols == model_cols if keys_present else False
        if not columns_the_same:
            self.logger.warning(
                f"val_data columns {data_cols} are not matching original features model was trained on: {model_cols}"
            )

        return keys_present and columns_the_same

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs):
        predictor: TabularPredictor = args.model
        val_data = args.val_data
        problem_type = predictor.problem_type
        label = predictor.label
        y_true_val, y_pred_val, highest_error, undecided = self._predict(problem_type, predictor, val_data)
        test_data = args.test_data
        test_data_present = args.test_data is not None and label in args.test_data.columns

        y_true_test = None
        y_pred_test = None
        if test_data_present:
            test_data = args.test_data
            y_true_test, y_pred_test, highest_error, undecided = self._predict(problem_type, predictor, test_data)

        _data = test_data if test_data_present else val_data
        importance = predictor.feature_importance(_data.reset_index(drop=True), silent=True)
        leaderboard = predictor.leaderboard(_data, silent=True)

        labels = predictor.class_labels
        s = {
            "problem_type": predictor.problem_type,
            "importance": importance,
            "leaderboard": leaderboard,
            "labels": labels,
            "y_true_val": y_true_val,
            "y_pred_val": y_pred_val,
        }

        try:
            y_pred_train = args.model.get_oof_pred_proba()
            s["y_true_train"] = args.train_data[args.label]
            s["y_pred_train"] = y_pred_train
        except AssertionError:
            # OOF is not available - don't use it
            pass

        if test_data_present:
            s["y_true_test"] = y_true_test
            s["y_pred_test"] = y_pred_test

        if undecided is not None:
            s["undecided"] = undecided
        if highest_error is not None:
            s["highest_error"] = highest_error

        if problem_type in [BINARY, MULTICLASS]:
            cm = confusion_matrix(y_true_val, y_pred_val, normalize=self.normalize, labels=labels)
            s["confusion_matrix_normalized"] = self.normalize is not None
            s["confusion_matrix"] = cm

        state.model_evaluation = s

    def _predict(self, problem_type, predictor, val_data):
        label = predictor.label
        y_true_val = val_data[label]
        y_pred_val = predictor.predict(val_data)
        highest_error = None
        undecided = None
        if predictor.problem_type in [BINARY, MULTICLASS]:
            y_proba = predictor.predict_proba(val_data)

            misclassified = y_proba[y_true_val != y_pred_val]
            expected_value = misclassified.join(y_true_val).apply(lambda row: row.loc[row[label]], axis=1)
            predicted_value = misclassified.max(axis=1)
            highest_error = predicted_value - expected_value
            highest_error.name = "error"

            scores = np.sort(misclassified.values, axis=1)
            diff = scores[:, -1] - expected_value
            undecided = pd.Series(index=y_pred_val.index, data=diff, name="error").sort_values(ascending=True)
            undecided = val_data.join(y_proba).join(undecided).sort_values(by="error")
            highest_error = (
                val_data.join(y_proba, rsuffix="_pred")
                .join(highest_error, how="inner")
                .sort_values(by="error", ascending=False)
            )
        elif problem_type == REGRESSION:
            highest_error = np.abs(y_pred_val - y_true_val).sort_values(ascending=False)
            highest_error.name = "error"
            highest_error = (
                val_data.join(y_pred_val, rsuffix="_pred")
                .join(highest_error, how="inner")
                .sort_values(by="error", ascending=False)
            )
        return y_true_val, y_pred_val, highest_error, undecided
