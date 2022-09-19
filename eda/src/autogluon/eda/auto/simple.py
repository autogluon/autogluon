from ..analysis import BaseAnalysis, Sampler, RawTypesAnalysis, FeatureInteraction
from ..visualization import SimpleLinearLayout, FeatureInteractionVisualization


def analyze(train_data=None, test_data=None, val_data=None, model=None, label=None, sample=None, anlz_facets=[], viz_facets=[], return_state=False, state=None):
    a = BaseAnalysis(
        train_data=train_data,
        test_data=test_data,
        val_data=val_data,
        model=model,
        label=label,
        children=[
            Sampler(sample=sample, children=anlz_facets),
        ]
    )

    if state is not None:
        a.state = state

    state = a.fit()

    SimpleLinearLayout(
        state=state,
        facets=viz_facets,
    ).render()

    if return_state:
        return state


def analyze_interaction(x=None, y=None, hue=None, viz_args={}, fig_args={}, **analysis_args):
    analyze(**analysis_args, anlz_facets=[
        RawTypesAnalysis(),
        FeatureInteraction(x=x, y=y, hue=hue),
    ], viz_facets=[
        FeatureInteractionVisualization(fig_args=fig_args, **viz_args),
    ])
