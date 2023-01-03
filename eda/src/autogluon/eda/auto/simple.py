import logging
from typing import Any, Dict, List, Optional, Union

from autogluon.common.utils.log_utils import verbosity2loglevel

from .. import AnalysisState
from ..analysis import FeatureInteraction
from ..analysis.base import AbstractAnalysis, BaseAnalysis
from ..analysis.dataset import RawTypesAnalysis, Sampler
from ..visualization import FeatureInteractionVisualization
from ..visualization.base import AbstractVisualization
from ..visualization.layouts import SimpleVerticalLinearLayout

__all__ = ["analyze", "analyze_interaction"]


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
