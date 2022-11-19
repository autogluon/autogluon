from typing import Union, List, Optional

from sklearn.metrics import confusion_matrix

from autogluon.core.constants import BINARY, MULTICLASS
from autogluon.tabular import TabularPredictor
from .base import AnalysisState, AbstractAnalysis

__all__ = ["AutoGluonModelEvaluator"]


class AutoGluonModelEvaluator(AbstractAnalysis):
    """
    Evaluate AutoGluon model performance.

    This analysis requires trained classifier passed in `model` arg and uses 'val_data' dataset to asses model performance.

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
        return "model" in args and "val_data" in args

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs):
        predictor: TabularPredictor = args.model
        val_data = args.val_data
        problem_type = predictor.problem_type
        label = predictor.label
        y_true = val_data[label]
        y_pred = predictor.predict(val_data)
        importance = predictor.feature_importance(val_data.reset_index(drop=True))

        s = {
            "problem_type": predictor.problem_type,
            "y_true": y_true,
            "y_pred": y_pred,
            "importance": importance,
        }
        if problem_type in [BINARY, MULTICLASS]:
            cm = confusion_matrix(y_true, y_pred, normalize=self.normalize, labels=y_true.unique())
            s["confusion_matrix_normalized"] = self.normalize is not None
            s["confusion_matrix"] = cm

        state.model_evaluation = s
