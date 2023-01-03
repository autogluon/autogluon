import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from autogluon.common.utils.log_utils import verbosity2loglevel

from .. import AnalysisState
from ..analysis import AutoGluonModelEvaluator, AutoGluonModelQuickFit, FeatureInteraction, TrainValidationSplit
from ..analysis.base import AbstractAnalysis, BaseAnalysis
from ..analysis.dataset import RawTypesAnalysis, Sampler
from ..visualization import (
    ConfusionMatrix,
    FeatureImportance,
    FeatureInteractionVisualization,
    MarkdownSectionComponent,
    ModelLeaderboard,
    RegressionEvaluation,
)
from ..visualization.base import AbstractVisualization
from ..visualization.layouts import SimpleVerticalLinearLayout

__all__ = ["analyze", "analyze_interaction", "quick_fit"]


def analyze(
    train_data=None,
    test_data=None,
    val_data=None,
    model=None,
    label: Optional[str] = None,
    state: Union[None, dict, AnalysisState] = None,
    sample: Union[None, int, float] = None,
    anlz_facets: Optional[List[AbstractAnalysis]] = None,
    viz_facets: Optional[List[AbstractVisualization]] = None,
    return_state: bool = False,
    verbosity: int = 2,
):
    """
    This helper creates `BaseAnalysis` wrapping passed analyses into
    `Sampler` if needed, then fits and renders produced state with
    specified visualizations.

    Parameters
    ----------
    train_data
        training dataset
    test_data
        test dataset
    val_data
        validation dataset
    model
        trained `Predictor`
    label: str
        target variable
    state: Union[None, dict, AnalysisState], default = None
        pass prior state if necessary; the object will be updated during `anlz_facets` `fit` call.
    sample: Union[None, int, float], default = None
        sample size; if `int`, then row number is used;
        `float` must be between 0.0 and 1.0 and represents fraction of dataset to sample;
        `None` means no sampling
        See also :func:`autogluon.eda.analysis.dataset.Sampler`
    anlz_facets: List[AbstractAnalysis]
        analyses to add to this composite analysis
    viz_facets: List[AbstractVisualization]
        visualizations to add to this composite analysis
    return_state: bool, default = False
        return state if `True`
    verbosity: int, default = 2,
        Verbosity levels range from 0 to 4 and control how much information is printed.
        Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
        If using logging, you can alternatively control amount of information printed via `logger.setLevel(L)`,
        where `L` ranges from 0 to 50 (Note: higher values of `L` correspond to fewer print statements, opposite of verbosity levels).

    Returns
    -------
        state after `fit` call if `return_state` is `True`; `None` otherwise

    """

    if viz_facets is None:
        viz_facets = []

    if anlz_facets is None:
        anlz_facets = []

    if state is not None:
        assert isinstance(state, (dict, AnalysisState))

    if not isinstance(state, AnalysisState):
        state = AnalysisState(state)

    root_logger = logging.getLogger("autogluon")
    root_log_level = root_logger.level
    log_level = verbosity2loglevel(verbosity)
    root_logger.setLevel(log_level)

    analysis = BaseAnalysis(
        state=state,
        train_data=train_data,
        test_data=test_data,
        val_data=val_data,
        model=model,
        label=label,
        children=[
            Sampler(sample=sample, children=anlz_facets),
        ],
    )

    state = analysis.fit()

    SimpleVerticalLinearLayout(
        facets=viz_facets,
    ).render(state)

    root_logger.setLevel(root_log_level)  # Reset log level

    if return_state:
        return state


def analyze_interaction(
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    viz_args: Optional[Dict[str, Any]] = None,
    fig_args: Optional[Dict[str, Any]] = None,
    **analysis_args,
):
    """
    This helper performs simple feature interaction analysis.

    Parameters
    ----------
    x: Optional[str], default = None
    y: Optional[str], default = None
    hue: Optional[str], default = None
    viz_args: Optional[dict], default = None
        kwargs to pass into visualization component
    fig_args: Optional[Dict[str, Any]], default = None,
        kwargs to pass into chart figure

    Examples
    --------
    >>> import pandas as pd
    >>> import autogluon.eda.auto as auto
    >>>
    >>> df_train = pd.DataFrame(...)
    >>>
    >>> auto.analyze_interaction(x='Age', hue='Survived', train_data=df_train, viz_args=dict(headers=True, alpha=0.2))

    See Also
    --------
    :py:class:`~autogluon.eda.analysis.interaction.FeatureInteraction`
    :py:class:`~autogluon.eda.visualization.interaction.FeatureInteractionVisualization`

    """
    if viz_args is None:
        viz_args = {}

    if fig_args is None:
        fig_args = {}

    key = "__analysis__"
    return analyze(
        **analysis_args,
        anlz_facets=[
            RawTypesAnalysis(),
            FeatureInteraction(key=key, x=x, y=y, hue=hue),
        ],
        viz_facets=[
            FeatureInteractionVisualization(key=key, fig_args=fig_args, **viz_args),
        ],
    )


def quick_fit(
    train_data: pd.DataFrame,
    label: str,
    path: Optional[str] = None,
    val_size: float = 0.3,
    problem_type: str = "auto",
    sample: Union[None, int, float] = None,
    state: Union[None, dict, AnalysisState] = None,
    return_state: bool = False,
    verbosity: int = 0,
    show_feature_importance_barplots: bool = False,
    **fit_args,
):
    """
    This helper performs quick model fit analysis and then produces a composite report of the results.

    The analysis is structured in a sequence of operations:
        - Sample if `sample` is specified.
        - Perform train-test split using `val_size` ratio
        - Fit AutoGluon estimator given `fit_args`; if `hyperparameters` not present in args, then use default ones
            (Random Forest by default - because it is interpretable)
        - Display report

    The reports include:
        - confusion matrix for classification problems; predictions vs actual for regression problems
        - model leaderboard
        - feature importance

    Parameters
    ----------
    train_data: DataFrame
        training dataset
    label: str
        target variable
    path: Optional[str], default = None,
        path for models saving
    problem_type: str, default = 'auto'
        problem type to use. Valid problem_type values include ['auto', 'binary', 'multiclass', 'regression', 'quantile', 'softclass']
        auto means it will be Auto-detected using AutoGluon methods.
    sample: Union[None, int, float], default = None
        sample size; if `int`, then row number is used;
        `float` must be between 0.0 and 1.0 and represents fraction of dataset to sample;
        `None` means no sampling
        See also :func:`autogluon.eda.analysis.dataset.Sampler`
    val_size: float, default = 0.3
        fraction of training set to be assigned as validation set during the split.
    state: Union[None, dict, AnalysisState], default = None
        pass prior state if necessary; the object will be updated during `anlz_facets` `fit` call.
    return_state: bool, default = False
        return state if `True`
    verbosity: int, default = 0
        Verbosity levels range from 0 to 4 and control how much information is printed.
        Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
        If using logging, you can alternatively control amount of information printed via `logger.setLevel(L)`,
        where `L` ranges from 0 to 50 (Note: higher values of `L` correspond to fewer print statements, opposite of verbosity levels).
    show_feature_importance_barplots: bool, default = False
        if `True`, then barplot char will ba added with feature importance visualization
    fit_args
        kwargs to pass into `TabularPredictor` fit

    Returns
    -------
        state after `fit` call if `return_state` is `True`; `None` otherwise

    See Also
    --------
    :py:class:`~autogluon.eda.visualization.model.ConfusionMatrix`
    :py:class:`~autogluon.eda.visualization.model.RegressionEvaluation`
    :py:class:`~autogluon.eda.visualization.model.ModelLeaderboard`
    :py:class:`~autogluon.eda.visualization.model.FeatureImportance`

    """
    if state is not None:
        assert isinstance(state, (dict, AnalysisState))

    if not isinstance(state, AnalysisState):
        state = AnalysisState(state)

    if "hyperparameters" not in fit_args:
        fit_args = fit_args.copy()
        fit_args["hyperparameters"] = {
            "RF": [
                {
                    "criterion": "entropy",
                    "max_depth": 15,
                    "ag_args": {"name_suffix": "Entr", "problem_types": ["binary", "multiclass"]},
                },
                {
                    "criterion": "squared_error",
                    "max_depth": 15,
                    "ag_args": {"name_suffix": "MSE", "problem_types": ["regression", "quantile"]},
                },
            ],
        }

    return analyze(
        train_data=train_data,
        label=label,
        sample=sample,
        state=state,
        return_state=return_state,
        anlz_facets=[
            TrainValidationSplit(
                val_size=val_size,
                problem_type=problem_type,
                children=[
                    AutoGluonModelQuickFit(
                        estimator_args={"path": path},
                        verbosity=verbosity,
                        problem_type=problem_type,
                        children=[
                            AutoGluonModelEvaluator(),
                        ],
                        **fit_args,
                    ),
                ],
            )
        ],
        viz_facets=[
            MarkdownSectionComponent(markdown=f"### Model Prediction for {label}"),
            ConfusionMatrix(fig_args=dict(figsize=(3, 3)), annot_kws={"size": 12}),
            RegressionEvaluation(fig_args=dict(figsize=(6, 6)), marker="o", scatter_kws={"s": 5}),
            MarkdownSectionComponent(markdown="### Model Leaderboard"),
            ModelLeaderboard(),
            MarkdownSectionComponent(markdown="### Feature Importance for Trained Model"),
            FeatureImportance(show_barplots=show_feature_importance_barplots),
        ],
    )
