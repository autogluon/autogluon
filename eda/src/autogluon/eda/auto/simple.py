from ..analysis.base import BaseAnalysis
from ..analysis.dataset import Sampler, RawTypesAnalysis
from ..analysis.interaction import FeatureInteraction
from ..visualization.interaction import FeatureInteractionVisualization
from ..visualization.layouts import SimpleVerticalLinearLayout


def analyze(train_data=None, test_data=None, val_data=None, model=None, label=None, sample=None, anlz_facets=[], viz_facets=[], return_state=False, state=None):
    a = BaseAnalysis(
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

    state = a.fit()

    SimpleVerticalLinearLayout(
        facets=viz_facets,
    ).render(state)

    if return_state:
        return state


def analyze_interaction(x=None, y=None, hue=None, viz_args={}, fig_args={}, **analysis_args):
    analyze(**analysis_args, anlz_facets=[
        RawTypesAnalysis(),
        FeatureInteraction(x=x, y=y, hue=hue),
    ], viz_facets=[
        FeatureInteractionVisualization(fig_args=fig_args, **viz_args),
    ])
