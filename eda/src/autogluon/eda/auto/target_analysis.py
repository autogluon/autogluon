from copy import deepcopy

from . import analyze
from ..analysis import RawTypesAnalysis, FeatureInteraction, Correlation, Namespace, DatasetSummary, MissingValuesAnalysis, VariableTypeAnalysis, \
    DistributionFit
from ..visualization import FeatureInteractionVisualization, CorrelationVisualization, DatasetStatistics
from ..visualization.layouts import MarkdownSectionComponent


def target_analysis(train_data, label=None, sample=None,
                    correlation_method='spearman', correlations_focus_threshold=0.5,
                    fit_args={}, fig_args={}, viz_args={},
                    **kwargs):
    corr_args = {**dict(method=correlation_method, focus_field=label, focus_field_threshold=correlations_focus_threshold), **fit_args.get('correlation', {})}
    state = analyze(train_data=train_data, label=label, sample=sample, return_state=True, anlz_facets=[
        Namespace(namespace='target_statistics', train_data=train_data[[label]], children=[
            RawTypesAnalysis(),
            VariableTypeAnalysis(),
            FeatureInteraction(x=label),
            DatasetSummary(),
            MissingValuesAnalysis(),
            DistributionFit(columns=label),
        ]),
        Namespace(namespace='target_correlations', train_data=train_data, children=[
            RawTypesAnalysis(),
            VariableTypeAnalysis(),
            Correlation(**corr_args),
        ]),

    ], viz_facets=[
        MarkdownSectionComponent(f'## Target Variable analysis: {label}'),
        FeatureInteractionVisualization(
            namespace='target_statistics', fig_args=fig_args.get('target_distribution', {}), **viz_args.get('target_distribution', {})
        ),
        DatasetStatistics(namespace='target_statistics', headers=False, **viz_args.get('dataset_statistics', {})),
        CorrelationVisualization(namespace='target_correlations', fig_args=fig_args.get('correlation', {}), headers=True, **viz_args.get('correlation', {})),
    ])

    high_corr_vars = {var: corr for var, corr in state.target_correlations.correlations_focus_high_corr.train_data[label].items()}

    # Prototype object for FeatureInteraction visualization
    feature_viz = FeatureInteractionVisualization(
        namespace='target_correlations', fig_args=fig_args.get('feature_interaction', {}), **viz_args.get('feature_interaction', {})
    )

    anlz_facets = []
    viz_facets = []
    for var, corr in high_corr_vars.items():
        if var == label:
            continue

        viz_facets.append(MarkdownSectionComponent(f'### Variable with high correlation vs target - {var}: {corr:.2f}'))
        for c in [FeatureInteraction(x=var), FeatureInteraction(x=var, y=label)]:
            anlz_facets.append(c)
            viz_facets.append(deepcopy(feature_viz))
        viz_facets.append(MarkdownSectionComponent('---'))

    # Only render one component at-a-time
    for idx, f in enumerate([vf for vf in viz_facets if isinstance(vf, FeatureInteractionVisualization)]):
        f.render_only_idx = idx

    analyze(train_data=train_data, state=state, anlz_facets=[
        Namespace(namespace='target_correlations', children=[
            *anlz_facets,
        ])
    ], viz_facets=[
        *viz_facets,
    ])
