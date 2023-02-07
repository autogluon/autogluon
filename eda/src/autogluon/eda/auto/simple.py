import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from autogluon.common.utils.log_utils import verbosity2loglevel

from .. import AnalysisState
from ..analysis import (
    ApplyFeatureGenerator,
    AutoGluonModelEvaluator,
    AutoGluonModelQuickFit,
    Correlation,
    DistributionFit,
    FeatureInteraction,
    MissingValuesAnalysis,
    ProblemTypeControl,
    TrainValidationSplit,
    XShiftDetector,
)
from ..analysis.base import AbstractAnalysis, BaseAnalysis
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
from ..utils.defaults import QuickFitDefaults
from ..visualization import (
    ConfusionMatrix,
    CorrelationVisualization,
    DatasetStatistics,
    DatasetTypeMismatch,
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
from ..visualization.interaction import FeatureDistanceAnalysisVisualization
from ..visualization.layouts import SimpleVerticalLinearLayout

__all__ = [
    "analyze",
    "analyze_interaction",
    "covariate_shift_detection",
    "dataset_overview",
    "missing_values_analysis",
    "quick_fit",
    "target_analysis",
]


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

    return state if return_state else None


def analyze_interaction(
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
    x: Optional[str], default = None
    y: Optional[str], default = None
    hue: Optional[str], default = None
    fit_distributions: Union[bool, str, List[str]], default = False,
        If `True`, or list of distributions is provided, then fit distributions. Performed only if `y` and `hue` are not present.
    chart_args: Optional[dict], default = None
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
    >>> auto.analyze_interaction(x='Age', hue='Survived', train_data=df_train, chart_args=dict(headers=True, alpha=0.2))

    See Also
    --------
    :py:class:`~autogluon.eda.analysis.interaction.FeatureInteraction`
    :py:class:`~autogluon.eda.visualization.interaction.FeatureInteractionVisualization`
    """
    assert (
        (x is not None) or (y is not None) or (hue is not None)
    ), "At least one of the parameters must be specified: x, y or hue"
    fig_args = get_empty_dict_if_none(fig_args).copy()
    if "figsize" not in fig_args:
        fig_args["figsize"] = (12, 6)

    chart_args = get_empty_dict_if_none(chart_args)

    key = "__analysis__"

    _analysis_args = analysis_args.copy()
    _analysis_args.pop("return_state", None)
    state: AnalysisState = analyze(return_state=True, **_analysis_args, anlz_facets=[RawTypesAnalysis(), VariableTypeAnalysis()])  # type: ignore

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

            analysis_facets.append(DistributionFit(columns=x, keep_top_n=5, distributions_to_fit=dists))  # type: ignore # x is always present

    _analysis_args = analysis_args.copy()
    _analysis_args.pop("state", None)

    return analyze(
        **_analysis_args,
        state=state,
        anlz_facets=analysis_facets,
        viz_facets=[
            FeatureInteractionVisualization(key=key, fig_args=fig_args, **chart_args),
        ],
    )


def _is_single_numeric_variable(x, y, hue, x_type):
    return (x is not None) and (y is None) and (hue is None) and (x_type == "numeric")


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
    fig_args: Optional[Dict[str, Dict[str, Any]]] = None,
    chart_args: Optional[Dict[str, Dict[str, Any]]] = None,
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
        - confusion_matrix - confusion matrix chart for classification predictor
        - regression_eval - regression predictor results chart
        - feature_importance - feature importance barplot chart


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
    fig_args: Optional[Dict[str, Any]], default = None,
        figures args for vizualizations; key == component; value = dict of kwargs for component figure
    chart_args: Optional[Dict[str, Any]], default = None,
        figures args for vizualizations; key == component; value = dict of kwargs for component chart

    Returns
    -------
        state after `fit` call if `return_state` is `True`; `None` otherwise

    Examples
    --------
    >>> import autogluon.eda.analysis as eda
    >>>
    >>> # Quick fit
    >>> state = auto.quick_fit(
    >>>     train_data=..., label=...,
    >>>     return_state=True,  # return state object from call
    >>>     save_model_to_state=True,  # store fitted model into the state
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

    fig_args = get_empty_dict_if_none(fig_args)
    chart_args = get_empty_dict_if_none(chart_args)

    fit_args = get_default_estimator_if_not_specified(fit_args)

    return analyze(
        train_data=train_data,
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
                        estimator_args={"path": path},
                        verbosity=verbosity,
                        problem_type=problem_type,
                        children=[
                            AutoGluonModelEvaluator(),
                        ],
                        **fit_args,
                    ),
                ],
            ),
        ],
        viz_facets=[
            MarkdownSectionComponent(markdown=f"### Model Prediction for {label}"),
            ConfusionMatrix(
                fig_args=fig_args.get("confusion_matrix", {}),
                **chart_args.get("confusion_matrix", dict(annot_kws={"size": 12})),
            ),
            RegressionEvaluation(
                fig_args=fig_args.get("regression_eval", {}),
                **chart_args.get("regression_eval", dict(marker="o", scatter_kws={"s": 5})),
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
        ],
    )


def dataset_overview(
    train_data: Optional[pd.DataFrame] = None,
    test_data: Optional[pd.DataFrame] = None,
    val_data: Optional[pd.DataFrame] = None,
    label: Optional[str] = None,
    state: Union[None, dict, AnalysisState] = None,
    return_state: bool = False,
    sample: Union[None, int, float] = None,
    fig_args: Optional[Dict[str, Dict[str, Any]]] = None,
    chart_args: Optional[Dict[str, Dict[str, Any]]] = None,
):
    """
    Shortcut to perform high-level datasets summary overview (counts, frequencies, missing statistics, types info).

    Supported `fig_args`/`chart_args` keys:
        - feature_distance - feature distance dendrogram chart


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
    sample: Union[None, int, float], default = None
        sample size; if `int`, then row number is used;
        `float` must be between 0.0 and 1.0 and represents fraction of dataset to sample;
        `None` means no sampling
        See also :func:`autogluon.eda.analysis.dataset.Sampler`
    fig_args: Optional[Dict[str, Any]], default = None,
        figures args for vizualizations; key == component; value = dict of kwargs for component figure
    chart_args: Optional[Dict[str, Any]], default = None,
        figures args for vizualizations; key == component; value = dict of kwargs for component chart

    Examples
    --------
    >>> import autogluon.eda.analysis as eda
    >>>
    >>> auto.dataset_overview(
    >>>     train_data=df_train, test_data=df_test, label=target_col,
    >>>     chart_args={'feature_distance': dict(orientation='left')},
    >>>     fig_args={'feature_distance': dict(figsize=(6,6))},
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

    fig_args = get_empty_dict_if_none(fig_args)
    chart_args = get_empty_dict_if_none(chart_args)

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
                interactions.append(MarkdownSectionComponent(f"Feature interaction between `{nodes[0]}`/`{n}`"))
                interactions.append(FeatureInteractionVisualization(key=f"{nodes[0]}:{n}"))

            analyze(
                train_data=train_data,
                state=state,
                anlz_facets=[FeatureInteraction(key=f"{nodes[0]}:{n}", x=nodes[0], y=n) for n in nodes[1:]],
                viz_facets=[
                    MarkdownSectionComponent(
                        f'**Near duplicate group analysis: `{"`, `".join(nodes)}` - distance `{group["distance"]:.4f}`**'
                    ),
                    *interactions,
                ],
            )

    return state if return_state else None


def covariate_shift_detection(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    label: str,
    sample: Union[None, int, float] = None,
    path: Optional[str] = None,
    state: Union[None, dict, AnalysisState] = None,
    return_state: bool = False,
    verbosity: int = 0,
    **fit_args,
):
    """
    Shortcut for covariate shift detection analysis.

    Detects a change in covariate (X) distribution between training and test, which we call XShift.  It can tell you
    if your training set is not representative of your test set distribution.  This is done with a Classifier 2
    Sample Test.

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
    sample: Union[None, int, float], default = None
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

    state = analyze(
        train_data=train_data,
        test_data=test_data,
        label=label,
        sample=sample,
        state=state,
        return_state=True,
        anlz_facets=[
            XShiftDetector(classifier_kwargs=dict(path=path, verbosity=verbosity), classifier_fit_kwargs=fit_args)
        ],
        viz_facets=[XShiftSummary()],
    )

    # Plot distribution differences between datasets
    xshift_results: AnalysisState = state.xshift_results  # type: ignore # state is always present
    if xshift_results.detection_status:
        fi = xshift_results.feature_importance
        fi = fi[fi.p_value <= xshift_results.pvalue_threshold]
        vars_to_plot = fi.index.tolist()
        if len(vars_to_plot) > 0:
            _train_data = train_data[vars_to_plot].copy()
            _train_data["__dataset__"] = "train_data"
            _test_data = test_data[vars_to_plot].copy()
            _test_data["__dataset__"] = "test_data"
            df_all = pd.concat([_train_data, _test_data], ignore_index=True)

            for var in vars_to_plot:
                pvalue = fi.loc[var]["p_value"]
                analyze(
                    viz_facets=[
                        MarkdownSectionComponent(
                            f"**`{var}` values distribution between datasets; p-value: `{pvalue:.4f}`**"
                        )
                    ]
                )
                analyze_interaction(train_data=df_all, state=state, x=var, hue="__dataset__")

    return state if return_state else None


def _is_lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa

        return True
    except (ImportError, OSError):
        return False


def get_default_estimator_if_not_specified(fit_args):
    if ("hyperparameters" not in fit_args) and ("presets" not in fit_args):
        fit_args = fit_args.copy()

        fit_args["fit_weighted_ensemble"] = False
        if _is_lightgbm_available():
            fit_args["hyperparameters"] = QuickFitDefaults.DEFAULT_LGBM_CONFIG
        else:
            fit_args["hyperparameters"] = QuickFitDefaults.DEFAULT_RF_CONFIG
    return fit_args


def get_empty_dict_if_none(value) -> dict:
    if value is None:
        value = {}
    return value


def target_analysis(
    train_data: pd.DataFrame,
    label: str,
    test_data: Optional[pd.DataFrame] = None,
    problem_type: str = "auto",
    fit_distributions: Union[bool, str, List[str]] = True,
    sample: Union[None, int, float] = None,
    state: Union[None, dict, AnalysisState] = None,
    return_state: bool = False,
) -> Optional[AnalysisState]:
    """
    Target variable composite analysis.

    Performs the following analysis components of the label field:
     - basic summary stats
     - feature values distribution charts; adds fitted distributions for numeric targets
     - target correlations analysis; with interaction charts of target vs high-correlated features

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
    sample: Union[None, int, float], default = None
        sample size; if `int`, then row number is used;
        `float` must be between 0.0 and 1.0 and represents fraction of dataset to sample;
        `None` means no sampling
        See also :func:`autogluon.eda.analysis.dataset.Sampler`
    return_state: bool, default = False
        return state if `True`

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
    )

    state = _render_distribution_fit_information_if_available(state, label)
    state = _render_correlation_analysis(state, train_data, label, sample)
    state = _render_features_highly_correlated_with_target(state, train_data, label, sample)

    return state if return_state else None


def _render_features_highly_correlated_with_target(state, train_data, label, sample) -> AnalysisState:
    fields = state.correlations_focus_high_corr.train_data.index.tolist()  # type: ignore
    analyze(
        train_data=train_data,
        state=state,
        sample=sample,
        return_state=True,
        anlz_facets=[FeatureInteraction(key=f"{f}:{label}", x=f, y=label) for f in fields],
        viz_facets=[FeatureInteractionVisualization(headers=True, key=f"{f}:{label}") for f in fields],
    )
    return state


def _render_correlation_analysis(state, train_data, label, sample) -> AnalysisState:
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
            CorrelationVisualization(headers=True),
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
                    dist_info.append(f'   - p-value: {p["pvalue"]:.3f}')
                    dist_info.append(f"   - Parameters: ({params})")
        else:
            dist_info.append(
                f" - ⚠️ none of the [attempted](https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions) "  # type: ignore
                f"distribution fits satisfy specified minimum p-value threshold: `{state.distributions_fit_pvalue_min}`"
            )
        analyze(viz_facets=[MarkdownSectionComponent("\n".join(dist_info))])
    return state


def missing_values_analysis(
    graph_type: str = "matrix",
    train_data: Optional[pd.DataFrame] = None,
    test_data: Optional[pd.DataFrame] = None,
    val_data: Optional[pd.DataFrame] = None,
    state: Union[None, dict, AnalysisState] = None,
    return_state: bool = False,
    sample: Union[None, int, float] = None,
    **chart_args,
):
    """
    Perform quick analysis of missing values across datasets.

    Parameters
    ----------
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
    train_data: Optional[DataFrame]
        training dataset
    test_data: Optional[DataFrame], default = None
        test dataset
    val_data
        validation dataset
    state: Union[None, dict, AnalysisState], default = None
        pass prior state if necessary; the object will be updated during `anlz_facets` `fit` call.
    return_state: bool, default = False
        return state if `True`
    sample: Union[None, int, float], default = None
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
