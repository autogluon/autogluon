from typing import Union, List, Optional

from .. import AnalysisState
from ..analysis.base import BaseAnalysis, AbstractAnalysis
from ..analysis.dataset import Sampler
from ..visualization.base import AbstractVisualization
from ..visualization.layouts import SimpleVerticalLinearLayout


def analyze(train_data=None,
            test_data=None,
            val_data=None,
            model=None,
            label: str = None,
            state: Union[None, dict, AnalysisState] = None,
            sample: Union[None, int, float] = None,
            anlz_facets: Optional[List[AbstractAnalysis]] = None,
            viz_facets: Optional[List[AbstractVisualization]] = None,
            return_state: bool = False):
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

    analysis = BaseAnalysis(
        state=state,
        train_data=train_data,
        test_data=test_data,
        val_data=val_data,
        model=model,
        label=label,
        children=[
            Sampler(sample=sample, children=anlz_facets),
        ]
    )

    state = analysis.fit()

    SimpleVerticalLinearLayout(
        facets=viz_facets,
    ).render(state)

    if return_state:
        return state
