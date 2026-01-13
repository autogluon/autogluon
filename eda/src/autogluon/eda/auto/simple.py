import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from autogluon.common.utils.log_utils import verbosity2loglevel
from autogluon.features import CategoryFeatureGenerator
from autogluon.tabular import TabularPredictor

from .. import AnalysisState
from ..analysis import (
    AnomalyDetectorAnalysis,
    ApplyFeatureGenerator,
    AutoGluonModelEvaluator,
    AutoGluonModelQuickFit,
    Correlation,
    DistributionFit,
    FeatureInteraction,
    MissingValuesAnalysis,
    ProblemTypeControl,
    ShapAnalysis,
    TrainValidationSplit,
    XShiftDetector,
)
from ..analysis.base import AbstractAnalysis, BaseAnalysis, SaveArgsToState
from ..analysis.dataset import (
    DatasetSummary,
    LabelInsightsAnalysis,
    RawTypesAnalysis,
    Sampler,
    SpecialTypesAnalysis,
    VariableTypeAnalysis,
)
from ..analysis.interaction import FeatureDistanceAnalysis
from ..state import is_key_present_in_state
from ..utils.common import expand_nested_args_into_nested_maps, get_empty_dict_if_none
from ..utils.defaults import QuickFitDefaults
from ..visualization import (
    AnomalyScoresVisualization,
    ConfusionMatrix,
    CorrelationVisualization,
    DatasetStatistics,
    DatasetTypeMismatch,
    ExplainForcePlot,
    ExplainWaterfallPlot,
    FeatureImportance,
    FeatureInteractionVisualization,
    LabelInsightsVisualization,
    MarkdownSectionComponent,
    MissingValues,
    ModelLeaderboard,
    PropertyRendererComponent,
    RegressionEvaluation,
    XShiftSummary,
)
from ..visualization.base import AbstractVisualization
from ..visualization.interaction import FeatureDistanceAnalysisVisualization, PDPInteractions
from ..visualization.layouts import SimpleVerticalLinearLayout

logger = logging.getLogger(__name__)

__all__ = [
    "analyze",
    "analyze_interaction",
    "covariate_shift_detection",
    "dataset_overview",
    "detect_anomalies",
    "explain_rows",
    "missing_values_analysis",
    "partial_dependence_plots",
    "quick_fit",
    "target_analysis",
]

DEFAULT_SAMPLE_SIZE = 10000


def analyze(
    train_data: Optional[pd.DataFrame] = None,
    test_data: Optional[pd.DataFrame] = None,
    val_data: Optional[pd.DataFrame] = None,
    model=None,
    label: Optional[str] = None,
    state: Union[None, dict, AnalysisState] = None,
    sample: Union[None, int, float] = DEFAULT_SAMPLE_SIZE,
    anlz_facets: Optional[List[AbstractAnalysis]] = None,
    viz_facets: Optional[List[AbstractVisualization]] = None,
    return_state: bool = False,
    verbosity: int = 2,
    **kwargs,
) -> Optional[AnalysisState]:
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
    sample: Union[None, int, float], default = 10000
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

    Examples
    --------
    >>> import autogluon.eda.analysis as eda
    >>> import autogluon.eda.visualization as viz
    >>> import autogluon.eda.auto as auto
    >>> state = auto.analyze(
    >>>     train_data=..., label=..., return_state=True,
    >>>     anlz_facets=[
    >>>         # Add analysis chain here
    >>>     ],
    >>>     viz_facets=[
    >>>         # Add visualization facets here
    >>>     ]
    >>> )

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

    return state if return_state else None


def analyze_interaction(
    train_data: pd.DataFrame,
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    fit_distributions: Union[bool, str, List[str]] = False,
    fig_args: Optional[Dict[str, Any]] = None,
    chart_args: Optional[Dict[str, Any]] = None,
    **analysis_args,
):
    """
    This helper performs simple feature interaction analysis.

    Parameters
    ----------
    train_data: pd.DataFrame
        training dataset
    x: Optional[str], default = None
    y: Optional[str], default = None
    hue: Optional[str], default = None
    fit_distributions: Union[bool, str, List[str]], default = False,
        If `True`, or list of distributions is provided, then fit distributions. Performed only if `y` and `hue` are not present.
    chart_args: Optional[dict], default = None
        kwargs to pass into visualization component
    fig_args: Optional[Dict[str, Any]], default = None,
        kwargs to pass into visualization component

    Examples
    --------
    >>> import pandas as pd
    >>> import autogluon.eda.auto as auto
    >>>
    >>> df_train = pd.DataFrame(...)
    >>>
    >>> auto.analyze_interaction(x='Age', hue='Survived', train_data=df_train, chart_args=dict(headers=True, alpha=0.2))

    See Also
    --------
    :py:class:`~autogluon.eda.analysis.interaction.FeatureInteraction`
    :py:class:`~autogluon.eda.visualization.interaction.FeatureInteractionVisualization`
    """
    assert (x is not None) or (y is not None) or (hue is not None), (
        "At least one of the parameters must be specified: x, y or hue"
    )
    fig_args = get_empty_dict_if_none(fig_args).copy()
    if "figsize" not in fig_args:
        fig_args["figsize"] = (12, 6)

    chart_args = get_empty_dict_if_none(chart_args)

    key = "__analysis__"

    _analysis_args = analysis_args.copy()
    _analysis_args.pop("return_state", None)

    pvalue_min = _analysis_args.pop("pvalue_min", 0.01)
    keep_top_n = _analysis_args.pop("keep_top_n", 5)
    numeric_as_categorical_threshold = _analysis_args.pop("numeric_as_categorical_threshold", 20)

    state: AnalysisState = analyze(
        train_data=train_data,
        return_state=True,
        **_analysis_args,
        anlz_facets=[
            RawTypesAnalysis(),
            VariableTypeAnalysis(numeric_as_categorical_threshold=numeric_as_categorical_threshold),
        ],
    )  # type: ignore

    analysis_facets: List[AbstractAnalysis] = [
        FeatureInteraction(key=key, x=x, y=y, hue=hue),
    ]

    if x is not None:
        x_type = state.variable_type.train_data[x]
        if _is_single_numeric_variable(x, y, hue, x_type) and (fit_distributions is not False):
            dists: Optional[List[str]]  # fit all
            if fit_distributions is True:
                dists = None
            elif isinstance(fit_distributions, str):
                dists = [fit_distributions]
            else:
                dists = fit_distributions

            analysis_facets.append(
                DistributionFit(columns=x, keep_top_n=keep_top_n, pvalue_min=pvalue_min, distributions_to_fit=dists)
            )  # type: ignore # x is always present

    _analysis_args = analysis_args.copy()
    _analysis_args.pop("state", None)

    return analyze(
        train_data=train_data,
        **_analysis_args,
        state=state,
        anlz_facets=analysis_facets,
        viz_facets=[
            FeatureInteractionVisualization(
                key=key,
                fig_args=fig_args,
                numeric_as_categorical_threshold=numeric_as_categorical_threshold,
                **chart_args,
            ),
        ],
    )


def _is_single_numeric_variable(x, y, hue, x_type):
    return (x is not None) and (y is None) and (hue is None) and (x_type == "numeric")


def quick_fit(
    train_data: pd.DataFrame,
    label: str,
    test_data: Optional[pd.DataFrame] = None,
    path: Optional[str] = None,
    val_size: float = 0.3,
    problem_type: str = "auto",
    fit_bagging_folds: int = 0,
    sample: Union[None, int, float] = DEFAULT_SAMPLE_SIZE,
    state: Union[None, dict, AnalysisState] = None,
    return_state: bool = False,
    save_model_to_state: bool = True,
    verbosity: int = 0,
    show_feature_importance_barplots: bool = False,
    estimator_args: Optional[Dict[str, Dict[str, Any]]] = None,
    fig_args: Optional[Dict[str, Dict[str, Any]]] = None,
    chart_args: Optional[Dict[str, Dict[str, Any]]] = None,
    render_analysis: bool = True,
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
        - samples with the highest prediction error - candidates for inspection
        - samples with the least distance from the other class - candidates for labeling

    Supported `fig_args`/`chart_args` keys:
        - `confusion_matrix.<property>` - confusion matrix chart for classification predictor
        - `regression_eval.<property>` - regression predictor results chart
        - `feature_importance.<property>` - feature importance barplot chart

    State attributes

    - `model`
        trained model
    - `model_evaluation.importance`
        feature importance calculated using the trained model
    - `model_evaluation.leaderboard`
        trained models leaderboard
    - `model_evaluation.highest_error`
        misclassified rows with the highest error between prediction and ground truth
    - `model_evaluation.undecided` (classification only)
        misclassified rows with the prediction closest to the decision boundary
    - `model_evaluation.confusion_matrix` (classification only)
        confusion matrix values

    Parameters
    ----------
    train_data: DataFrame
        training dataset
    test_data: DataFrame
        test dataset
    label: str
        target variable
    path: Optional[str], default = None,
        path for models saving
    problem_type: str, default = 'auto'
        problem type to use. Valid problem_type values include ['auto', 'binary', 'multiclass', 'regression', 'quantile', 'softclass']
        auto means it will be Auto-detected using AutoGluon methods.
    fit_bagging_folds: int, default = 0,
        shortcut to enable training with bagged folds; disabled if 0 (default)
    sample: Union[None, int, float], default = 10000
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
    save_model_to_state: bool, default = True,
        save fitted model into `state` under `model` key.
        This functionality might be helpful in cases when the fitted model could be usable for other purposes (i.e. imputers)
    verbosity: int, default = 0
        Verbosity levels range from 0 to 4 and control how much information is printed.
        Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
        If using logging, you can alternatively control amount of information printed via `logger.setLevel(L)`,
        where `L` ranges from 0 to 50 (Note: higher values of `L` correspond to fewer print statements, opposite of verbosity levels).
    show_feature_importance_barplots: bool, default = False
        if `True`, then barplot char will ba added with feature importance visualization
    estimator_args: Optional[Dict[str, Dict[str, Any]]], default = None,
        args to pass into the estimator constructor
    fit_args: Optional[Dict[str, Dict[str, Any]]], default = None,
        kwargs to pass into `TabularPredictor` fit.
    fig_args: Optional[Dict[str, Any]], default = None,
        figures args for visualizations; key == component; value = dict of kwargs for component figure. The args are supporting nested
        dot syntax: 'a.b.c'.
    chart_args: Optional[Dict[str, Any]], default = None,
        figures args for visualizations; key == component; value = dict of kwargs for component chart. The args are supporting nested
        dot syntax: 'a.b.c'.
    render_analysis: bool, default = True
        if `False`, then don't render any visualizations; this can be used if user just needs to train a model. It is recommended to use this option
        with `save_model_to_state=True` and `return_state=True` options.

    Returns
    -------
        state after `fit` call if `return_state` is `True`; `None` otherwise

    Examples
    --------
    >>> import autogluon.eda.analysis as eda
    >>> import autogluon.eda.auto as auto
    >>>
    >>> # Quick fit
    >>> state = auto.quick_fit(
    >>>     train_data=..., label=...,
    >>>     return_state=True,  # return state object from call
    >>>     fig_args={"regression_eval.figsize": (8,6)},  # customize regression evaluation `figsize`
    >>>     chart_args={"regression_eval.residuals_plot_mode": "hist"}  # customize regression evaluation `residuals_plot_mode`
    >>>     hyperparameters={'GBM': {}}  # train specific model
    >>> )
    >>>
    >>> # Using quick fit model
    >>> model = state.model
    >>> y_pred = model.predict(test_data)

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

    fig_args = expand_nested_args_into_nested_maps(get_empty_dict_if_none(fig_args))
    chart_args = expand_nested_args_into_nested_maps(get_empty_dict_if_none(chart_args))

    estimator_args = get_empty_dict_if_none(estimator_args)
    assert fit_bagging_folds >= 0, "fit_bagging_folds must be non-negative"
    fit_args = get_default_estimator_if_not_specified(fit_args, fit_bagging_folds)

    if "path" not in estimator_args:
        estimator_args["path"] = path  # type: ignore

    if (test_data is not None) and (label not in test_data.columns):
        test_data = None

    if render_analysis:
        viz = [
            MarkdownSectionComponent(markdown=f"### Model Prediction for {label}"),
            MarkdownSectionComponent(
                condition_fn=(lambda state: is_key_present_in_state(state, "model_evaluation.y_pred_test")),
                markdown="Using `test_data` for `Test` points",
            ),
            MarkdownSectionComponent(
                condition_fn=(lambda state: not is_key_present_in_state(state, "model_evaluation.y_pred_test")),
                markdown="Using validation data for `Test` points",
            ),
            ConfusionMatrix(
                fig_args=fig_args.get("confusion_matrix", {}),
                **chart_args.get("confusion_matrix", dict(annot_kws={"size": 12})),
            ),
            RegressionEvaluation(
                fig_args=fig_args.get("regression_eval", {}),
                **chart_args.get("regression_eval", {}),
            ),
            MarkdownSectionComponent(markdown="### Model Leaderboard"),
            ModelLeaderboard(),
            MarkdownSectionComponent(markdown="### Feature Importance for Trained Model"),
            FeatureImportance(
                show_barplots=show_feature_importance_barplots,
                fig_args=fig_args.get("feature_importance", {}),
                **chart_args.get("feature_importance", {}),
            ),
            MarkdownSectionComponent(markdown="### Rows with the highest prediction error"),
            MarkdownSectionComponent(markdown="Rows in this category worth inspecting for the causes of the error"),
            PropertyRendererComponent("model_evaluation.highest_error", transform_fn=(lambda df: df.head(10))),
            MarkdownSectionComponent(
                condition_fn=(lambda state: is_key_present_in_state(state, "model_evaluation.undecided")),
                markdown="### Rows with the least distance vs other class",
            ),
            MarkdownSectionComponent(
                condition_fn=(lambda state: is_key_present_in_state(state, "model_evaluation.undecided")),
                markdown="Rows in this category are the closest to the decision boundary vs the other class "
                "and are good candidates for additional labeling",
            ),
            PropertyRendererComponent("model_evaluation.undecided", transform_fn=(lambda df: df.head(10))),
        ]
    else:
        viz = []

    return analyze(
        train_data=train_data,
        test_data=test_data,
        label=label,
        sample=sample,
        state=state,
        return_state=return_state,
        anlz_facets=[
            ProblemTypeControl(problem_type=problem_type),
            TrainValidationSplit(
                val_size=val_size,
                children=[
                    AutoGluonModelQuickFit(
                        estimator_args=estimator_args,
                        verbosity=verbosity,
                        problem_type=problem_type,
                        save_model_to_state=save_model_to_state,
                        children=[
                            AutoGluonModelEvaluator(),
                        ],
                        **fit_args,
                    ),
                ],
            ),
        ],
        viz_facets=viz,
    )


def dataset_overview(
    train_data: Optional[pd.DataFrame] = None,
    test_data: Optional[pd.DataFrame] = None,
    val_data: Optional[pd.DataFrame] = None,
    label: Optional[str] = None,
    state: Union[None, dict, AnalysisState] = None,
    return_state: bool = False,
    sample: Union[None, int, float] = DEFAULT_SAMPLE_SIZE,
    fig_args: Optional[Dict[str, Dict[str, Any]]] = None,
    chart_args: Optional[Dict[str, Dict[str, Any]]] = None,
):
    """
    Shortcut to perform high-level datasets summary overview (counts, frequencies, missing statistics, types info).

    Supported `fig_args`/`chart_args` keys:
        - `feature_distance.<property>` - feature distance dendrogram chart
        - `chart.<variable>.<property>` - near-duplicate groups visualizations chart. If chart is labeled as a relationship <A>/<B>, then <variable> is <B>

    Parameters
    ----------
    train_data: Optional[DataFrame], default = None
        training dataset
    test_data: Optional[DataFrame], default = None
        test dataset
    val_data: Optional[DataFrame], default = None
        validation dataset
    label: : Optional[str], default = None
        target variable
    state: Union[None, dict, AnalysisState], default = None
        pass prior state if necessary; the object will be updated during `anlz_facets` `fit` call.
    return_state: bool, default = False
        return state if `True`
    sample: Union[None, int, float], default = 10000
        sample size; if `int`, then row number is used;
        `float` must be between 0.0 and 1.0 and represents fraction of dataset to sample;
        `None` means no sampling
        See also :func:`autogluon.eda.analysis.dataset.Sampler`
    fig_args: Optional[Dict[str, Any]], default = None,
        figures args for visualizations; key == component; value = dict of kwargs for component figure
    chart_args: Optional[Dict[str, Any]], default = None,
        figures args for visualizations; key == component; value = dict of kwargs for component chart

    Examples
    --------
    >>> import autogluon.eda.analysis as eda
    >>>
    >>> auto.dataset_overview(
    >>>     train_data=df_train, test_data=df_test, label=target_col,
    >>>     chart_args={'feature_distance.orientation': 'left'},
    >>>     fig_args={'feature_distance.figsize': (6,6)},
    >>> )

    See Also
    --------
    :py:class:`~autogluon.eda.analysis.dataset.DatasetSummary`
    :py:class:`~autogluon.eda.analysis.dataset.RawTypesAnalysis`
    :py:class:`~autogluon.eda.analysis.dataset.SpecialTypesAnalysis`
    :py:class:`~autogluon.eda.analysis.missing.MissingValuesAnalysis`
    :py:class:`~autogluon.eda.visualization.dataset.DatasetStatistics`
    :py:class:`~autogluon.eda.visualization.dataset.DatasetTypeMismatch`

    """

    fig_args = expand_nested_args_into_nested_maps(get_empty_dict_if_none(fig_args))
    chart_args = expand_nested_args_into_nested_maps(get_empty_dict_if_none(chart_args))

    state = analyze(
        train_data=train_data,
        test_data=test_data,
        val_data=val_data,
        label=label,
        sample=sample,
        state=state,
        return_state=True,
        anlz_facets=[
            DatasetSummary(),
            MissingValuesAnalysis(),
            RawTypesAnalysis(),
            VariableTypeAnalysis(),
            SpecialTypesAnalysis(),
            ApplyFeatureGenerator(category_to_numbers=True, children=[FeatureDistanceAnalysis()]),
        ],
        viz_facets=[
            DatasetStatistics(headers=True),
            DatasetTypeMismatch(headers=True),
            MarkdownSectionComponent("### Feature Distance"),
            FeatureDistanceAnalysisVisualization(
                fig_args=fig_args.get("feature_distance", {}), **chart_args.get("feature_distance", {})
            ),
        ],
    )

    # Groups analysis
    distance = state.feature_distance  # type: ignore # state is always present
    if len(distance.near_duplicates) > 0:  # type: ignore # state is always present
        for group in distance.near_duplicates:
            nodes = group["nodes"]

            interactions: List[AbstractVisualization] = []
            for n in nodes[1:]:
                if state.variable_type.train_data[n] != "category":  # type: ignore
                    interactions.append(MarkdownSectionComponent(f"Feature interaction between `{nodes[0]}`/`{n}`"))
                    interactions.append(
                        FeatureInteractionVisualization(
                            key=f"{nodes[0]}:{n}",
                            fig_args=fig_args.get("chart", {}).get(n, {}),
                            **chart_args.get("chart", {}).get(n, {}),
                        )
                    )

            analyze(
                train_data=train_data,
                state=state,
                anlz_facets=[FeatureInteraction(key=f"{nodes[0]}:{n}", x=nodes[0], y=n) for n in nodes[1:]],
                viz_facets=[
                    *interactions,
                ],
            )

    return state if return_state else None


def covariate_shift_detection(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    label: str,
    sample: Union[None, int, float] = DEFAULT_SAMPLE_SIZE,
    path: Optional[str] = None,
    state: Union[None, dict, AnalysisState] = None,
    return_state: bool = False,
    verbosity: int = 0,
    fig_args: Optional[Dict[str, Any]] = None,
    chart_args: Optional[Dict[str, Any]] = None,
    **fit_args,
):
    """
    Shortcut for covariate shift detection analysis.

    Detects a change in covariate (X) distribution between training and test, which we call XShift.  It can tell you
    if your training set is not representative of your test set distribution.  This is done with a Classifier 2
    Sample Test.

    Supported `fig_args`/`chart_args` keys:
        - `chart.<variable_name>.<property>` - properties for charts rendered during the analysis

    Parameters
    ----------
    train_data: Optional[DataFrame]
        training dataset
    test_data: Optional[DataFrame]
        test dataset
    label: : Optional[str]
        target variable
    state: Union[None, dict, AnalysisState], default = None
        pass prior state if necessary; the object will be updated during `anlz_facets` `fit` call.
    sample: Union[None, int, float], default = 10000
        sample size; if `int`, then row number is used;
        `float` must be between 0.0 and 1.0 and represents fraction of dataset to sample;
        `None` means no sampling
        See also :func:`autogluon.eda.analysis.dataset.Sampler`
    path: Optional[str], default = None,
        path for models saving
    return_state: bool, default = False
        return state if `True`
    verbosity: int, default = 0
        Verbosity levels range from 0 to 4 and control how much information is printed.
        Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
        If using logging, you can alternatively control amount of information printed via `logger.setLevel(L)`,
        where `L` ranges from 0 to 50 (Note: higher values of `L` correspond to fewer print statements, opposite of verbosity levels).
    fit_args
        kwargs to pass into `TabularPredictor` fit
    fig_args: Optional[Dict[str, Any]], default = None,
        figures args for visualizations; key == component; value = dict of kwargs for component figure. The args are supporting nested
        dot syntax: 'a.b.c'. Charts args are following the convention of `<variable_name>.<param>`
        (i.e. `chart.PassengerId.figsize` will result in setting `figsize` on `PassengerId` figure.
    chart_args: Optional[Dict[str, Any]], default = None,
        figures args for visualizations; key == component; value = dict of kwargs for component chart. The args are supporting nested
        dot syntax: 'a.b.c'. Charts args are following the convention of `<variable_name>.<param>`
        (i.e. `chart.PassengerId.fill` will result in setting `fill` on `PassengerId` chart.

    Returns
    -------
        state after `fit` call if `return_state` is `True`; `None` otherwise

    Examples
    --------
    >>> import autogluon.eda.auto as auto
    >>>
    >>> # use default settings
    >>> auto.covariate_shift_detection(train_data=..., test_data=..., label=...)
    >>>
    >>> # customize classifier and verbosity level
    >>> auto.covariate_shift_detection(train_data=..., test_data=..., label=..., verbosity=2, hyperparameters = {'GBM': {}})

    See Also
    --------
    :py:class:`~autogluon.eda.analysis.shift.XShiftDetector`
    :py:class:`~autogluon.eda.visualization.shift.XShiftSummary`

    """
    fit_args = get_default_estimator_if_not_specified(fit_args)
    fig_args = expand_nested_args_into_nested_maps(get_empty_dict_if_none(fig_args))
    chart_args = expand_nested_args_into_nested_maps(get_empty_dict_if_none(chart_args))

    state = analyze(
        train_data=train_data,
        test_data=test_data,
        label=label,
        sample=sample,
        state=state,
        return_state=True,
        anlz_facets=[
            RawTypesAnalysis(),
            VariableTypeAnalysis(),
            XShiftDetector(classifier_kwargs=dict(path=path, verbosity=verbosity), classifier_fit_kwargs=fit_args),
        ],
        viz_facets=[XShiftSummary()],
    )

    # Plot distribution differences between datasets
    # TODO: move `vars_to_plot` calculation to analysis
    # type: ignore # state is always present
    xshift: AnalysisState = state.xshift_results  # type: ignore[union-attr]  # state is not none
    if xshift.detection_status:
        vars_to_plot = xshift.shift_features[: XShiftSummary.MAX_FEATURES_TO_DISPLAY]
        if len(vars_to_plot) > 0:
            _train_data = train_data[vars_to_plot].copy()
            _train_data["__dataset__"] = "train_data"
            _test_data = test_data[vars_to_plot].copy()
            _test_data["__dataset__"] = "test_data"
            df_all = pd.concat([_train_data, _test_data], ignore_index=True)

            for var in vars_to_plot:
                if state.variable_type.train_data[var] != "category":  # type: ignore
                    pvalue = xshift.feature_importance.loc[var]["p_value"]
                    analyze(
                        viz_facets=[
                            MarkdownSectionComponent(
                                f"**`{var}` values distribution between datasets; p-value: `{pvalue:.4f}`**"
                            )
                        ]
                    )

                    analyze_interaction(
                        train_data=df_all,
                        state=state,
                        x=var,
                        hue="__dataset__",
                        fig_args=fig_args.get("chart", {}).get(var, {}),
                        chart_args=chart_args.get("chart", {}).get(var, {}),
                    )

    return state if return_state else None


def _is_lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa

        return True
    except (ImportError, OSError):
        return False


def get_default_estimator_if_not_specified(fit_args, fit_bagging_folds: int = 0):
    if ("hyperparameters" not in fit_args) and ("presets" not in fit_args):
        fit_args = fit_args.copy()

        fit_args["fit_weighted_ensemble"] = False
        if fit_bagging_folds > 0:
            fit_args = {**dict(num_bag_folds=fit_bagging_folds, num_bag_sets=1, num_stack_levels=0), **fit_args}
            if ("ag_args_ensemble" not in fit_args) or ("fold_fitting_strategy" not in fit_args["ag_args_ensemble"]):
                fit_args["ag_args_ensemble"] = {"fold_fitting_strategy": "sequential_local"}
        if _is_lightgbm_available():
            fit_args["hyperparameters"] = QuickFitDefaults.DEFAULT_LGBM_CONFIG
        else:
            fit_args["hyperparameters"] = QuickFitDefaults.DEFAULT_RF_CONFIG
    return fit_args


def target_analysis(
    train_data: pd.DataFrame,
    label: str,
    test_data: Optional[pd.DataFrame] = None,
    problem_type: str = "auto",
    fit_distributions: Union[bool, str, List[str]] = True,
    sample: Union[None, int, float] = DEFAULT_SAMPLE_SIZE,
    state: Union[None, dict, AnalysisState] = None,
    return_state: bool = False,
    fig_args: Optional[Dict[str, Any]] = None,
    chart_args: Optional[Dict[str, Any]] = None,
) -> Optional[AnalysisState]:
    """
    Target variable composite analysis.

    Performs the following analysis components of the label field:
     - basic summary stats
     - feature values distribution charts; adds fitted distributions for numeric targets
     - target correlations analysis; with interaction charts of target vs high-correlated features

    Supported `fig_args`/`chart_args` keys:
        - `correlation.<property>` - properties for correlation heatmap
        - `chart.<variable_name>.<property>` - properties for charts rendered during the analysis.
        If <variable_name> is matching `label` value, then this will modify the top chart; all other values will be affecting label/<variable_name>
        interaction charts

    Parameters
    ----------
    train_data: Optional[DataFrame]
        training dataset
    test_data: Optional[DataFrame], default = None
        test dataset
    label: : Optional[str]
        target variable
    problem_type: str, default = 'auto'
        problem type to use. Valid problem_type values include ['auto', 'binary', 'multiclass', 'regression', 'quantile', 'softclass']
        auto means it will be Auto-detected using AutoGluon methods.
    fit_distributions: Union[bool, str, List[str]], default = False,
        If `True`, or list of distributions is provided, then fit distributions. Performed only if `y` and `hue` are not present.
    state: Union[None, dict, AnalysisState], default = None
        pass prior state if necessary; the object will be updated during `anlz_facets` `fit` call.
    sample: Union[None, int, float], default = 10000
        sample size; if `int`, then row number is used;
        `float` must be between 0.0 and 1.0 and represents fraction of dataset to sample;
        `None` means no sampling
        See also :func:`autogluon.eda.analysis.dataset.Sampler`
    return_state: bool, default = False
        return state if `True`
    fig_args: Optional[Dict[str, Any]], default = None,
        figures args for visualizations; key == component; value = dict of kwargs for component figure. The args are supporting nested
        dot syntax: 'a.b.c'. Charts args are following the convention of `<variable_name>.<param>`
        (i.e. `chart.PassengerId.figsize` will result in setting `figsize` on `<target>`/`PassengerId` figure.
    chart_args: Optional[Dict[str, Any]], default = None,
        figures args for visualizations; key == component; value = dict of kwargs for component chart. The args are supporting nested
        dot syntax: 'a.b.c'. Charts args are following the convention of `<variable_name>.<param>`
        (i.e. `chart.PassengerId.fill` will result in setting `fill` on `<target>`/`PassengerId` chart.

    Returns
    -------
    state after `fit` call if `return_state` is `True`; `None` otherwise

    Examples
    --------
    >>> import autogluon.eda.analysis as eda
    >>>
    >>> auto.target_analysis(train_data=..., label=...)

    """

    assert label in train_data.columns, f"label `{label}` is not in `train_data` columns: `{train_data.columns}`"

    fig_args = expand_nested_args_into_nested_maps(get_empty_dict_if_none(fig_args))
    chart_args = expand_nested_args_into_nested_maps(get_empty_dict_if_none(chart_args))

    if (test_data is not None) and (label in test_data.columns):
        _test_data = test_data[[label]]
    else:
        _test_data = None

    # Basic variable information table
    state: AnalysisState = analyze(  # type: ignore # state is always present
        train_data=train_data[[label]],
        test_data=_test_data,
        label=label,
        state=state,
        sample=sample,
        return_state=True,
        anlz_facets=[
            DatasetSummary(),
            MissingValuesAnalysis(),
            RawTypesAnalysis(),
            SpecialTypesAnalysis(),
            ProblemTypeControl(problem_type=problem_type),
            LabelInsightsAnalysis(),
        ],
        viz_facets=[
            MarkdownSectionComponent("## Target variable analysis"),
            MarkdownSectionComponent(
                "### Label Insights",
                condition_fn=(lambda s: is_key_present_in_state(s, "label_insights")),
            ),
            LabelInsightsVisualization(),
            DatasetStatistics(),
        ],
    )

    # Distribution chart
    state = analyze_interaction(
        train_data=train_data,
        sample=sample,
        x=label,
        state=state,
        return_state=True,
        fit_distributions=fit_distributions,
        fig_args=fig_args.get("chart", {}).get(label, {}),
        chart_args=chart_args.get("chart", {}).get(label, {}),
    )

    state = _render_distribution_fit_information_if_available(state, label)
    state = _render_correlation_analysis(state, train_data, label, sample, fig_args, chart_args)
    state = _render_features_highly_correlated_with_target(state, train_data, label, sample, fig_args, chart_args)

    return state if return_state else None


def _render_features_highly_correlated_with_target(
    state, train_data, label, sample, fig_args, chart_args
) -> AnalysisState:
    fields = state.correlations_focus_high_corr.train_data.index.tolist()  # type: ignore
    analyze(
        train_data=train_data,
        state=state,
        sample=sample,
        return_state=True,
        anlz_facets=[FeatureInteraction(key=f"{f}:{label}", x=f, y=label) for f in fields],
        viz_facets=[
            FeatureInteractionVisualization(
                headers=True,
                key=f"{f}:{label}",
                fig_args=fig_args.get("chart", {}).get(f, {}),
                **chart_args.get("chart", {}).get(f, {}),
            )
            for f in fields
        ],
    )
    return state


def _render_correlation_analysis(state, train_data, label, sample, fig_args, chart_args) -> AnalysisState:
    state = analyze(
        train_data=train_data,
        sample=sample,
        state=state,
        return_state=True,
        label=label,
        anlz_facets=[ApplyFeatureGenerator(category_to_numbers=True, children=[Correlation(focus_field=label)])],
    )
    corr_info = ["### Target variable correlations"]
    if len(state.correlations_focus_high_corr.train_data) < 1:  # type: ignore
        corr_info.append(
            f" - ⚠️ no fields with absolute correlation greater than "  # type: ignore
            f"`{state.correlations_focus_field_threshold}` found for target variable `{label}`."
        )
    analyze(
        state=state,
        viz_facets=[
            MarkdownSectionComponent("\n".join(corr_info)),
            CorrelationVisualization(
                headers=True, fig_args=fig_args.get("correlation", {}), **chart_args.get("correlation", {})
            ),
        ],
    )
    return state


def _render_distribution_fit_information_if_available(state, label) -> Optional[AnalysisState]:
    if state.distributions_fit is not None:  # type: ignore # state is always present
        dist_fit_state = state.distributions_fit.train_data  # type: ignore
        dist_info = ["### Distribution fits for target variable"]
        if (label in dist_fit_state) and (len(dist_fit_state[label]) > 0):
            for d, p in state.distributions_fit.train_data[label].items():  # type: ignore
                dist_info.append(
                    f" - [{d}](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.{d}.html)"
                )
                if p.param is not None and len(p.param) > 0:
                    params = ", ".join([f"{shape}: {param}" for shape, param in zip(p.shapes, p.param)])
                    dist_info.append(f"   - p-value: {p['pvalue']:.3f}")
                    dist_info.append(f"   - Parameters: ({params})")
        else:
            dist_info.append(
                f" - ⚠️ none of the [attempted](https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions) "  # type: ignore
                f"distribution fits satisfy specified minimum p-value threshold: `{state.distributions_fit_pvalue_min}`"
            )
        analyze(viz_facets=[MarkdownSectionComponent("\n".join(dist_info))])
    return state


def missing_values_analysis(
    train_data: Optional[pd.DataFrame] = None,
    test_data: Optional[pd.DataFrame] = None,
    val_data: Optional[pd.DataFrame] = None,
    graph_type: str = "matrix",
    state: Union[None, dict, AnalysisState] = None,
    return_state: bool = False,
    sample: Union[None, int, float] = DEFAULT_SAMPLE_SIZE,
    **chart_args,
):
    """
    Perform quick analysis of missing values across datasets.

    Parameters
    ----------
    train_data: Optional[DataFrame]
        training dataset
    test_data: Optional[DataFrame], default = None
        test dataset
    val_data
        validation dataset
    graph_type: str, default = 'matrix'
        One of the following visualization types:
        - matrix - nullity matrix is a data-dense display which lets you quickly visually pick out patterns in data completion
            This visualization will comfortably accommodate up to 50 labelled variables.
            Past that range labels begin to overlap or become unreadable, and by default large displays omit them.
        - bar - visualizes how many rows are non-null vs null in the column. Logarithmic scale can by specifying `log=True` in `kwargs`
        - heatmap - correlation heatmap measures nullity correlation: how strongly the presence or absence of one
            variable affects the presence of another. Nullity correlation ranges from -1
            (if one variable appears the other definitely does not) to 0 (variables appearing or not appearing have no effect on one another)
            to 1 (if one variable appears the other definitely also does).
            Entries marked <1 or >-1 have a correlation that is close to being exactingly negative or positive but is still not quite perfectly so.
        - dendrogram - the dendrogram allows to more fully correlate variable completion, revealing trends deeper than the pairwise ones
            visible in the correlation heatmap. The dendrogram uses a hierarchical clustering algorithm (courtesy of scipy) to bin variables
            against one another by their nullity correlation (measured in terms of binary distance).
            At each step of the tree the variables are split up based on which combination minimizes the distance of the remaining clusters.
            The more monotone the set of variables, the closer their total distance is to zero, and the closer their average distance (the y-axis) is to zero.
    state: Union[None, dict, AnalysisState], default = None
        pass prior state if necessary; the object will be updated during `anlz_facets` `fit` call.
    return_state: bool, default = False
        return state if `True`
    sample: Union[None, int, float], default = 10000
        sample size; if `int`, then row number is used;
        `float` must be between 0.0 and 1.0 and represents fraction of dataset to sample;
        `None` means no sampling
        See also :func:`autogluon.eda.analysis.dataset.Sampler`

    Returns
    -------
    state after `fit` call if `return_state` is `True`; `None` otherwise

    Examples
    --------
    >>> import autogluon.eda.auto as auto
    >>>
    >>> auto.missing_values_analysis(train_data=...)

    See Also
    --------
    :py:class:`~autogluon.eda.analysis.missing.MissingValuesAnalysis`
    :py:class:`~autogluon.eda.visualization.dataset.DatasetStatistics`
    :py:class:`~autogluon.eda.visualization.missing.MissingValues`

    """
    # TODO add null equivalents: i.e. >50% of values are the same (i.e. 0 is frequently used as null equivalent)
    return analyze(
        train_data=train_data,
        test_data=test_data,
        val_data=val_data,
        state=state,
        return_state=return_state,
        sample=sample,
        anlz_facets=[
            MissingValuesAnalysis(),
        ],
        viz_facets=[
            MarkdownSectionComponent("### Missing Values Analysis"),
            DatasetStatistics(),
            MissingValues(graph_type=graph_type, **chart_args),
        ],
    )


def explain_rows(
    train_data: pd.DataFrame,
    model: TabularPredictor,
    rows: pd.DataFrame,
    positive_class: Optional = None,
    display_rows: bool = False,
    plot: Optional[str] = "force",
    baseline_sample: int = 100,
    return_state: bool = False,
    fit_args: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Optional[AnalysisState]:
    """
    Kernel SHAP is a method that uses a special weighted linear regression
    to compute the importance of each feature. The computed importance values
    are Shapley values from game theory and also coefficients from a local linear
    regression values analysis for the given rows.

    The results are rendered either as force plot or waterfall plot.

    Parameters
    ----------
    train_data: DataFrame
        training dataset
    model: TabularPredictor
        trained AutoGluon predictor
    rows: pd.DataFrame,
        rows to explain
    positive_class: Optional
        Optionally specify positive class to explain; if not provided, the value will be autodetected.
        For binary it's derived from `model.positive_class`.
        For multiclass it's the last column in prediction probabilities.
    display_rows: bool, default = False
        if `True` then display the row before the explanation chart
    plot: Optional[str], default = 'force'
        type of plot to visualize the Shapley values. Supported keys:
        - `force` - Visualize the given SHAP values with an additive force layout
        - `waterfall` - Visualize the given SHAP values with a waterfall layout
        - `None` - do not use any visualization
    baseline_sample: int, default = 100
        The background dataset size to use for integrating out features. To determine the impact
        of a feature, that feature is set to "missing" and the change in the model output
        is observed.
    return_state: bool, default = False
        return state if `True`
    fit_args: Optional[Dict[str, Any]], default = None,
        kwargs for `ShapAnalysis`.
    kwargs

    Examples
    --------
    >>> import autogluon.eda.auto as auto
    >>>
    >>> state = auto.quick_fit(
    >>>     train_data=...,
    >>>     label=...,
    >>>     return_state=True,
    >>> )
    >>>
    >>> # quick_fit stored model in `state.model`, and can be passed here.
    >>> # This will visualize 1st row of rows with the highest errors;
    >>> # these rows are stored under `state.model_evaluation.highest_error`
    >>> auto.explain_rows(
    >>>     train_data=...,
    >>>     model=state.model,
    >>>     display_rows=True,
    >>>     rows=state.model_evaluation.highest_error[:1],
    >>>     plot='waterfall',  # visualize as waterfall plot
    >>> )

    See Also
    --------
    :py:class:`~shap.KernelExplainer`
    :py:class:`~autogluon.eda.analysis.explain.ShapAnalysis`
    :py:class:`~autogluon.eda.visualization.explain.ExplainForcePlot`
    :py:class:`~autogluon.eda.visualization.explain.ExplainWaterfallPlot`
    """

    if fit_args is None:
        fit_args = {}

    if plot is None:
        viz_facets = None
    else:
        supported_plots = {
            "force": ExplainForcePlot,
            "waterfall": ExplainWaterfallPlot,
        }
        viz_cls = supported_plots.get(plot, None)
        assert viz_cls is not None, (
            f"plot must be one of the following values: {','.join(supported_plots.keys())}. "
            f"If no visualization required, then `None` can be passed."
        )
        viz_facets = [viz_cls(display_rows=display_rows, **kwargs)]

    return analyze(
        train_data=train_data[model.original_features],
        model=model,
        return_state=return_state,
        anlz_facets=[ShapAnalysis(rows, positive_class=positive_class, baseline_sample=baseline_sample, **fit_args)],  # type: ignore
        viz_facets=viz_facets,  # type: ignore
    )


def partial_dependence_plots(
    train_data: pd.DataFrame,
    label: str,
    target: Optional[Any] = None,
    features: Optional[Union[str, List[str]]] = None,
    two_way: bool = False,
    path: Optional[str] = None,
    max_ice_lines: int = 300,
    sample: Optional[Union[int, float]] = DEFAULT_SAMPLE_SIZE,
    fig_args: Optional[Dict[str, Dict[str, Any]]] = None,
    chart_args: Optional[Dict[str, Dict[str, Any]]] = None,
    show_help_text: bool = True,
    return_state: bool = False,
    col_number_warning: int = 20,
    **fit_args,
):
    """
    Partial Dependence Plot (PDP)

    Analyze and interpret the relationship between a target variable and a specific feature in a machine learning model.
    PDP helps in understanding the marginal effect of a feature on the predicted outcome while holding other features constant

    The visualizations have two modes:
    - Display Partial Dependence Plots (PDP) with Individual Conditional Expectation (ICE) - this is the default mode of operation
    - Two-Way PDP plots - this mode can be selected via passing two `features` and setting `two_way = True`

    ICE plots complement PDP by showing the relationship between a feature and the model's output for each individual instance in the dataset.
    ICE lines (blue) can be overlaid on PDPs (red) to provide a more detailed view of how the model behaves for specific instances.
    Here are some points on how to interpret PDPs with ICE lines:

    - `Central tendency`
        The PDP line represents the average prediction for different values of the feature of interest.
        Look for the overall trend of the PDP line to understand the average effect of the feature on the model's output.
    - `Variability`
        The ICE lines represent the predicted outcomes for individual instances as the feature of interest changes.
        Examine the spread of ICE lines around the PDP line to understand the variability in predictions for different instances.
    - `Non-linear relationships`
        Look for any non-linear patterns in the PDP and ICE lines.
        This may indicate that the model captures a non-linear relationship between the feature and the model's output.
    - `Heterogeneity`
        Check for instances where ICE lines have widely varying slopes, indicating different relationships between the feature and
        the model's output for individual instances. This may suggest interactions between the feature of interest and other features.
    - `Outliers`
        Look for any ICE lines that are very different from the majority of the lines.
        This may indicate potential outliers or instances that have unique relationships with the feature of interest.
    - `Confidence intervals`
        If available, examine the confidence intervals around the PDP line. Wider intervals may indicate a less certain relationship
        between the feature and the model's output, while narrower intervals suggest a more robust relationship.
    - `Interactions`
        By comparing PDPs and ICE plots for different features, you may detect potential interactions between features.
        If the ICE lines change significantly when comparing two features, this might suggest an interaction effect.

    Two-way PDP can visualize potential interactions between any two features. Here are a few cases when two-way PDP can give good results:

    - `Suspected interactions`: Even if two features are not highly correlated, they may still interact in the context of the model.
        If you suspect that there might be interactions between any two features, two-way PDP can help to verify the hypotheses.
    - `Moderate to high correlation`: If two features have a moderate to high correlation,
        a two-way PDP can show how the combined effect of these features influences the model's predictions.
        In this case, the plot can help reveal whether the relationship between the features is additive, multiplicative, or more complex.
    - `Complementary features`: If two features provide complementary information, a two-way PDP can help illustrate how the joint effect
        of these features impacts the model's predictions.
        For example, if one feature measures the length of an object and another measures its width, a two-way PDP could show how the
        combination of these features affects the predicted outcome.
    - `Domain knowledge`: If domain knowledge suggests that the relationship between two features might be important for the model's output,
        a two-way PDP can help to explore and validate these hypotheses.
    - `Feature importance`: If feature importance analysis ranks both features high in the leaderboard, it might be beneficial
        to examine their joint effect on the model's predictions.

    State attributes

    - `pdp_id_to_category_mappings`
        Categorical are represented in charts as numbers; id to value mappings are available in this property.

    Parameters
    ----------
    train_data: DataFrame
        training dataset
    label: str
        target variable
    target: Optional[Any], default = None
        In a multiclass setting, specifies the class for which the PDPs should be computed.
        Ignored in binary classification or classical regression settings
    features: Optional[Union[str, List[str]]], default = None
        feature subset to display; `None` means all features will be rendered.
    two_way: bool, default = False
        render two-way PDP; this mode works only when two `features` are specified
    path: Optional[str], default = None
        location to store the model trained for this task
    max_ice_lines: int, default = 300
        max number of ice lines to display for each sub-plot
    sample: Union[None, int, float], default = 10000
        sample size; if `int`, then row number is used;
        `float` must be between 0.0 and 1.0 and represents fraction of dataset to sample;
        `None` means no sampling
        See also :func:`autogluon.eda.analysis.dataset.Sampler`
    fig_args: Optional[Dict[str, Any]], default = None
        kwargs to pass into chart figure
    chart_args: Optional[dict], default = None
        kwargs to pass into visualization component
    show_help_text:bool, default = True
        if `True` shows additional information how to interpret the data
    return_state: bool, default = False
        return state if `True`
    col_number_warning: int, default = 20
        number of features to visualize after which the warning will be displayed to warn about rendering time
    fit_args: Optional[Dict[str, Dict[str, Any]]], default = None,
        kwargs to pass into `TabularPredictor` fit.

    Returns
    -------
    state after `fit` call if `return_state` is `True`; `None` otherwise

    Examples
    --------
    >>> import autogluon.eda.auto as auto
    >>>
    >>> # Plot all features in a grid
    >>> auto.partial_dependence_plots(train_data=..., label=...)
    >>>
    >>> # Plot two-way feature interaction for features `feature_a` and `feature_b`
    >>> auto.partial_dependence_plots(train_data=..., label=..., features=['feature_a', 'feature_b'], two_way=True)

    See Also
    --------
    :py:class:`~autogluon.eda.visualization.interaction.PDPInteractions`
    """

    chart_args, fig_args, features = _validate_and_normalize_pdp_args(
        train_data, features, fig_args, chart_args, col_number_warning
    )
    pdp_data, state, id_to_category_mappings = _prepare_pdp_data(train_data, label, sample, features)

    state = quick_fit(
        path=path,
        train_data=state.pdp_train_data,
        label=label,
        return_state=True,
        render_analysis=False,
        sample=sample,
        **fit_args,
    )
    state.pdp_data = pdp_data
    state.pdp_id_to_category_mappings = id_to_category_mappings

    if features is None:
        features = state.model_evaluation.importance.index.tolist()

    if len(id_to_category_mappings) > 0:
        cats = ", ".join([f"`{c}`" for c in id_to_category_mappings.keys()])
    else:
        cats = ""

    analyze(
        model=state.model,
        state=state,
        viz_facets=[
            MarkdownSectionComponent("### Two-way Partial Dependence Plots", condition_fn=lambda _: two_way),
            MarkdownSectionComponent(
                "Two-Way partial dependence plots (PDP) are useful for visualizing the relationship between a pair of features and the predicted outcome "
                "in a machine learning model. There are several things to look for when exploring the two-way plot:\n\n"
                "* **Shape of the interaction**: Look at the shape of the plot to understand the nature of the interaction. "
                "It could be linear, non-linear, or more complex.\n"
                "* **Feature value range**: Observe the range of the feature values on both axes to understand the domain of the interaction. "
                "This can help you identify whether the model is making predictions within reasonable bounds or if there are extrapolations "
                "beyond the training data.\n"
                "* **Areas of high uncertainty**: Look for areas in the plot where the predictions are less certain, which may be indicated by "
                "larger confidence intervals, higher variance, or fewer data points. These areas may require further investigation or additional data.\n"
                "* **Outliers and anomalies**: Check for any outliers or anomalies in the plot that may indicate issues with the model or the data. "
                "These could be regions of the plot with unexpected patterns or values that do not align with the overall trend.\n"
                "* **Sensitivity to feature values**: Assess how sensitive the predicted outcome is to changes in the feature values.\n\n"
                "<sub><sup>Use `show_help_text=False` to hide this information when calling this function.</sup></sub>",
                condition_fn=lambda _: show_help_text and two_way,
            ),
            MarkdownSectionComponent("### Partial Dependence Plots", condition_fn=lambda _: not two_way),
            MarkdownSectionComponent(
                "Individual Conditional Expectation (ICE) plots complement Partial Dependence Plots (PDP) by showing the "
                "relationship between a feature and the model's output for each individual instance in the dataset. ICE lines (blue) "
                "can be overlaid on PDPs (red) to provide a more detailed view of how the model behaves for specific instances. "
                "Here are some points on how to interpret PDPs with ICE lines:\n\n"
                "* **Central tendency**: The PDP line represents the average prediction for different values of the feature of interest. "
                "Look for the overall trend of the PDP line to understand the average effect of the feature on the model's output.\n"
                "* **Variability**: The ICE lines represent the predicted outcomes for individual instances as the feature of interest changes. "
                "Examine the spread of ICE lines around the PDP line to understand the variability in predictions for different instances.\n"
                "* **Non-linear relationships**: Look for any non-linear patterns in the PDP and ICE lines. This may indicate that the model "
                "captures a non-linear relationship between the feature and the model's output.\n"
                "* **Heterogeneity**: Check for instances where ICE lines have widely varying slopes, indicating different relationships "
                "between the feature and the model's output for individual instances. This may suggest interactions between the feature "
                "of interest and other features.\n"
                "* **Outliers**: Look for any ICE lines that are very different from the majority of the lines. This may indicate potential "
                "outliers or instances that have unique relationships with the feature of interest.\n"
                "* **Confidence** intervals: If available, examine the confidence intervals around the PDP line. Wider intervals may indicate "
                "a less certain relationship between the feature and the model's output, while narrower intervals suggest a more robust relationship.\n"
                "* **Interactions**: By comparing PDPs and ICE plots for different features, you may detect potential interactions between features. "
                "If the ICE lines change significantly when comparing two features, this might suggest an interaction effect.\n\n"
                "<sub><sup>Use `show_help_text=False` to hide this information when calling this function.</sup></sub>",
                condition_fn=lambda _: show_help_text and not two_way,
            ),
            PDPInteractions(
                features=features,
                two_way=two_way,
                fig_args=fig_args,
                sample=max_ice_lines,
                target=target,
                **chart_args,
            ),  # type: ignore
            MarkdownSectionComponent(
                f"The following variable(s) are categorical: {cats}. They are represented as the numbers in the figures above. "
                f"Mappings are available in `state.pdp_id_to_category_mappings`. The`state` can be returned from this call via adding `return_state=True`.",
                condition_fn=lambda _: len(id_to_category_mappings) > 0,
            ),
        ],
    )

    s = AnalysisState({"pdp_id_to_category_mappings": id_to_category_mappings})

    return s if return_state else None


def _validate_and_normalize_pdp_args(
    train_data: pd.DataFrame,
    features: Optional[Union[str, List[str]]] = None,
    fig_args: Optional[Dict[str, Dict[str, Any]]] = None,
    chart_args: Optional[Dict[str, Dict[str, Any]]] = None,
    col_number_warning: int = 20,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[List[str]]]:
    fig_args = get_empty_dict_if_none(fig_args)
    chart_args = get_empty_dict_if_none(chart_args)

    if features is not None:
        if type(features) is not list:
            features = [features]  # type: ignore
        features_not_present = [f for f in features if f not in train_data.columns]
        assert len(features_not_present) == 0, (
            f"Features {', '.join(features_not_present)} are not present in train_data: {', '.join(train_data.columns)}"
        )
    if features is None and len(train_data.columns) > col_number_warning:
        logger.warning(
            f"This visualization will render {len(train_data.columns)} charts. "
            f"This can take a while. This warning can be disabled by setting `col_number_warning` to a higher value."
        )
    return chart_args, fig_args, features


def _prepare_pdp_data(
    train_data: pd.DataFrame,
    label: str,
    sample: Optional[Union[int, float]] = None,
    features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, AnalysisState, Dict[str, Dict[int, str]]]:
    apply_gen = ApplyFeatureGenerator(
        category_to_numbers=True,
        children=[
            SaveArgsToState(
                params_mapping={
                    "train_data": "pdp_train_data",
                }
            )
        ],
    )
    state = analyze(
        train_data=train_data,
        label=label,
        return_state=True,
        anlz_facets=[apply_gen],
        sample=sample,
    )
    pdp_data = state.pdp_train_data  # type: ignore
    id_to_category_mappings: Dict[str, Dict[int, str]] = {}
    for gen in [item for sublist in apply_gen.feature_generator.generators for item in sublist]:
        if type(gen) is CategoryFeatureGenerator:
            id_to_category_mappings = {
                k: {i: v for i, v in enumerate(v.tolist())} for k, v in gen.category_map.items()
            }

    if features is not None:
        id_to_category_mappings = {k: v for k, v in id_to_category_mappings.items() if k in features}

    return pdp_data, state, id_to_category_mappings  # type: ignore


def detect_anomalies(
    train_data: pd.DataFrame,
    label: str,
    test_data: Optional[pd.DataFrame] = None,
    val_data: Optional[pd.DataFrame] = None,
    explain_top_n_anomalies: Optional[int] = None,
    show_top_n_anomalies: Optional[int] = 10,
    threshold_stds: float = 3,
    show_help_text: bool = True,
    state: Union[None, dict, AnalysisState] = None,
    sample: Union[None, int, float] = DEFAULT_SAMPLE_SIZE,
    return_state: bool = False,
    fig_args: Optional[Dict[str, Any]] = None,
    chart_args: Optional[Dict[str, Any]] = None,
    **anomaly_detector_kwargs,
) -> Optional[AnalysisState]:
    """
    Anomaly Detection

    This method is used to identify unusual patterns or behaviors in data that deviate significantly from the norm.
    It's best used when finding outliers, rare events, or suspicious activities that could indicate fraud, defects, or system failures.

    When interpreting anomaly scores, consider:

    - `Threshold`:
        Determine a suitable threshold to separate normal from anomalous data points, based on domain knowledge or statistical methods.
    - `Context`:
        Examine the context of anomalies, including time, location, and surrounding data points, to identify possible causes.
    - `False positives/negatives`:
        Be aware of the trade-offs between false positives (normal points classified as anomalies) and false negatives (anomalies missed).
    - `Feature relevance`:
        Ensure the features used for anomaly detection are relevant and contribute to the model's performance.
    - `Model performance`:
        Regularly evaluate and update the model to maintain its accuracy and effectiveness.

    It's important to understand the context and domain knowledge before deciding on an appropriate approach to deal with anomalies.
    The choice of method depends on the data's nature, the cause of anomalies, and the problem being addressed.
    The common ways to deal with anomalies:

    - `Removal`:
        If an anomaly is a result of an error, noise, or irrelevance to the analysis, it can be removed from the dataset
        to prevent it from affecting the model's performance.
    - `Imputation`:
        Replace anomalous values with appropriate substitutes, such as the mean, median, or mode of the feature,
        or by using more advanced techniques like regression or k-nearest neighbors.
    - `Transformation`:
        Apply transformations like log, square root, or z-score to normalize the data and reduce the impact of extreme values.
        Absolute dates might be transformed into relative features like age of the item.
    - `Capping`:
        Set upper and lower bounds for a feature, and replace values outside these limits with the bounds themselves.
        This method is also known as winsorizing.
    - `Separate modeling`:
        Treat anomalies as a distinct group and build a separate model for them, or use specialized algorithms designed
        for handling outliers, such as robust regression or one-class SVM.
    - `Incorporate as a feature`:
        Create a new binary feature indicating the presence of an anomaly, which can be useful if anomalies have predictive value.

    State attributes

    - `anomaly_detection.scores.<dataset>`
        scores for each of the datasets passed into analysis (i.e. `train_data`, `test_data`)
    - `state.anomaly_detection.anomalies.<dataset>`
        data points considered as anomalies - original rows with added `score` column sorted in descending score order.
        defined by `threshold_stds` parameter
    - `anomaly_detection.anomaly_score_threshold`
        anomaly score threshold above which data points are considered as anomalies;
        defined by `threshold_stds` parameter

    Parameters
    ----------
    train_data: DataFrame
        training dataset
    label: str
        target variable
    test_data: Optional[pd.DataFrame], default = None
        test dataset
    val_data: Optional[pd.DataFrame], default = None
        validation dataset
    explain_top_n_anomalies: Optional[int], default = None
        explain the anomaly scores for n rows with the highest scores; don't perform analysis if value is `None` or `0`
    show_top_n_anomalies: Optional[int], default = 10
        display n rows with highest anomaly scores
    threshold_stds: float, default = 3
        specifies how many standard deviations above mean anomaly score considered as anomalies
        (only needed for visualization, does not affect scores calculation)
    show_help_text:bool, default = True
        if `True` shows additional information how to interpret the data
    state: Union[None, dict, AnalysisState], default = None
        pass prior state if necessary; the object will be updated during `anlz_facets` `fit` call.
    sample: Union[None, int, float], default = 10000
        sample size; if `int`, then row number is used;
        `float` must be between 0.0 and 1.0 and represents fraction of dataset to sample;
        `None` means no sampling
        See also :func:`autogluon.eda.analysis.dataset.Sampler`
    return_state: bool, default = False
        return state if `True`
    fig_args: Optional[Dict[str, Any]], default = None,
        kwargs to pass into visualization component
    chart_args: Optional[dict], default = None
        kwargs to pass into visualization component
    anomaly_detector_kwargs
        kwargs to pass into :py:class:`~autogluon.eda.analysis.anomaly.AnomalyDetectorAnalysis`

    >>> import autogluon.eda.auto as auto
    >>>
    >>> state = auto.detect_anomalies(
    >>>     train_data=...,
    >>>     test_data=...,  # optional
    >>>     label=...,
    >>>     threshold_stds=3,
    >>>     show_top_n_anomalies=5,
    >>>     explain_top_n_anomalies=3,
    >>>     return_state=True,
    >>>     chart_args={
    >>>         'normal.color': 'lightgrey',
    >>>         'anomaly.color': 'orange',
    >>>     }
    >>> )
    >>>
    >>> # Getting anomaly scores from the analysis
    >>> train_anomaly_scores = state.anomaly_detection.scores.train_data
    >>> test_anomaly_scores = state.anomaly_detection.scores.test_data
    >>>
    >>> # Anomaly score threshold for specified level - see `threshold_stds` parameter
    >>> anomaly_score_threshold = state.anomaly_detection.anomaly_score_threshold

    Returns
    -------
    state after `fit` call if `return_state` is `True`; `None` otherwise

    See Also
    --------
    :py:class:`~autogluon.eda.analysis.anomaly.AnomalyDetectorAnalysis`
    :py:class:`~autogluon.eda.visualization.anomaly.AnomalyScoresVisualization`
    """
    fig_args_defaults = {"figsize": (12, 6)}
    fig_args = {**fig_args_defaults, **get_empty_dict_if_none(fig_args).copy()}

    chart_args = get_empty_dict_if_none(chart_args).copy()

    store_explainability_data = (explain_top_n_anomalies is not None) and explain_top_n_anomalies > 0
    _state: AnalysisState = analyze(  # type: ignore[assignment]  # always has value: return_state=True
        train_data=train_data,
        test_data=test_data,
        val_data=val_data,
        label=label,
        state=state,
        sample=sample,
        return_state=True,
        anlz_facets=[
            ProblemTypeControl(),
            ApplyFeatureGenerator(
                category_to_numbers=True,
                children=[
                    AnomalyDetectorAnalysis(
                        store_explainability_data=store_explainability_data, **anomaly_detector_kwargs
                    ),
                ],
            ),
        ],
    )

    analyze(
        state=_state,
        viz_facets=[
            MarkdownSectionComponent("### Anomaly Detection Report"),
            MarkdownSectionComponent(
                "When interpreting anomaly scores, consider:\n"
                "* **Threshold**: Determine a suitable threshold to separate normal from anomalous data points, "
                "    based on domain knowledge or statistical methods.\n"
                "* **Context**: Examine the context of anomalies, including time, location, and surrounding data points, to identify possible causes.\n"
                "* **False positives/negatives**: Be aware of the trade-offs between false positives (normal points classified as anomalies) "
                "    and false negatives (anomalies missed).\n"
                "* **Feature relevance**: Ensure the features used for anomaly detection are relevant and contribute to the model's performance.\n"
                "* **Model performance**: Regularly evaluate and update the model to maintain its accuracy and effectiveness.\n\n"
                "It's important to understand the context and domain knowledge before deciding on an appropriate approach to deal with anomalies."
                "The choice of method depends on the data's nature, the cause of anomalies, and the problem being addressed."
                "The common ways to deal with anomalies:\n\n"
                "* **Removal**: If an anomaly is a result of an error, noise, or irrelevance to the analysis, it can be removed from the dataset "
                "    to prevent it from affecting the model's performance.\n"
                "* **Imputation**: Replace anomalous values with appropriate substitutes, such as the mean, median, or mode of the feature,"
                "    or by using more advanced techniques like regression or k-nearest neighbors.\n"
                "* **Transformation**: Apply transformations like log, square root, or z-score to normalize the data and reduce the impact of extreme values.\n"
                "    Absolute dates might be transformed into relative features like age of the item.\n"
                "* **Capping**: Set upper and lower bounds for a feature, and replace values outside these limits with the bounds themselves."
                "    This method is also known as winsorizing.\n"
                "* **Separate modeling**: Treat anomalies as a distinct group and build a separate model for them, or use specialized algorithms designed"
                "    for handling outliers, such as robust regression or one-class SVM.\n"
                "* **Incorporate as a feature**: Create a new binary feature indicating the presence of an anomaly, "
                "    which can be useful if anomalies have predictive value.\n\n"
                "<sub><sup>Use `show_help_text=False` to hide this information when calling this function.</sup></sub>",
                condition_fn=lambda _: show_help_text,
            ),
            AnomalyScoresVisualization(threshold_stds=threshold_stds, headers=True, fig_args=fig_args, **chart_args),
        ],
    )

    # Store anomalies with the scores into the state
    _state.anomaly_detection.anomalies = {}
    anomaly_score_threshold = _state.anomaly_detection.scores.train_data.std() * threshold_stds
    _state.anomaly_detection.anomaly_score_threshold = anomaly_score_threshold
    for ds, df in AbstractAnalysis.available_datasets(
        AnalysisState({"train_data": train_data, "test_data": test_data, "val_data": val_data})
    ):
        anomaly_scores = _state.anomaly_detection.scores[ds]
        anomaly_idx = anomaly_scores[anomaly_scores >= anomaly_score_threshold].sort_values(ascending=False).index
        _state.anomaly_detection.anomalies[ds] = df.loc[anomaly_idx].join(anomaly_scores)

        if (show_top_n_anomalies is not None) and (show_top_n_anomalies > 0) and (len(anomaly_idx) > 0):
            analyze(
                state=_state,
                viz_facets=[
                    MarkdownSectionComponent(
                        markdown=f"**Top-{show_top_n_anomalies} `{ds}` anomalies (total: {len(anomaly_idx)})**"
                    ),
                    PropertyRendererComponent(
                        f"anomaly_detection.anomalies.{ds}", transform_fn=(lambda d: d.head(show_top_n_anomalies))
                    ),
                ],
            )

        if store_explainability_data:
            analyze(
                state=_state,
                viz_facets=[
                    MarkdownSectionComponent(
                        markdown="⚠️ Please note that the feature values shown on the charts below are transformed "
                        "into an internal representation; they may be encoded or modified based on internal preprocessing. "
                        "Refer to the original datasets for the actual feature values."
                    ),
                    MarkdownSectionComponent(
                        markdown="⚠️ The detector has seen this dataset; the may result in overly optimistic estimates. "
                        "Although the anomaly score in the explanation might not match, the magnitude of the feature scores "
                        "can still be utilized to evaluate the impact of the feature on the anomaly score.",
                        condition_fn=(lambda _: ds == "train_data"),  # noqa: B023
                    ),
                ],
            )

            explain_rows(
                **_state.anomaly_detection.explain_rows_fns[ds](anomaly_idx[:explain_top_n_anomalies]),
                plot="waterfall",
            )

    return _state if return_state else None
