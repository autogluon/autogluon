from ..analysis import BaseAnalysis, Sampler, RawTypesAnalysis, FeatureInteraction
from ..visualization import SimpleLinearLayout, FeatureInteractionVisualization


def analyze(train_data=None, test_data=None, label=None, sample=None, anlz_facets=[], viz_facets=[]):
    a = BaseAnalysis(
        train_data=train_data,
        test_data=test_data,
        label=label,
        children=[
            Sampler(sample=sample, children=anlz_facets),
        ]
    )

    state = a.fit()

    SimpleLinearLayout(
        state=state,
        facets=viz_facets,
    ).render()


def analyze_interaction(x=None, y=None, hue=None, viz_args={}, fig_args={}, **analysis_args):
    analyze(**analysis_args, anlz_facets=[
        RawTypesAnalysis(),
        FeatureInteraction(x=x, y=y, hue=hue),
    ], viz_facets=[
        FeatureInteractionVisualization(fig_args=fig_args, **viz_args),
    ])
