.. role:: hidden
    :class: hidden-section

Reference: Base APIs
====================

This section highlights the base APIs used by the EDA framework. The processing consists of the following parts:

1. Analysis graph construction - in this part the user constructs nested graph of analyses.

.. code-block:: python3

    analysis = BaseAnalysis(
        # State
        state=state,
        # Arguments
        train_data=train_data, test_data=test_data, val_data=val_data, model=model, label=label,
        # Nested analyses
        children=[
            Sampler(sample=sample, children=[
                DatasetSummary(),
                MissingValuesAnalysis(),
                RawTypesAnalysis(),
                SpecialTypesAnalysis(),
                ApplyFeatureGenerator(category_to_numbers=True, children=[
                    FeatureDistanceAnalysis()
                ]),
            ]),
        ],
    )

2. **.fit()** call to produce **State**, which is a nested dictionary that can be passed either for further processing
via next analysis or be rendered.

.. code-block:: python3

    state = analysis.fit()

3. Rendering: in this stage we construct components graph (a combination of layout components and visual components) and
then pass **State** generated previously as an input argument into **render()** call.

.. code-block:: python3

    viz = SimpleVerticalLinearLayout(
        facets=[
            DatasetStatistics(headers=True),
            DatasetTypeMismatch(headers=True),
            MarkdownSectionComponent("### Feature Distance"),
            FeatureDistanceAnalysisVisualization(),
        ],
    )
    viz.render(state)

Please note: it is possible that 1) components may depend on each other's output. There are two way they can do in analysis:
1) share values via state; 2) share values/shadow arguments (i.e. Sample component modifies train_data, test_data and val_data
arguments in the scope of calling children's fit().

autogluon.eda.analysis.base
---------------------------

.. automodule:: autogluon.eda.analysis.base
.. currentmodule:: autogluon.eda.analysis.base

.. autosummary::
   :nosignatures:

   AbstractAnalysis

:hidden:`AbstractAnalysis`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AbstractAnalysis
   :members:
   :inherited-members:

autogluon.eda.visualization.base
--------------------------------

.. automodule:: autogluon.eda.visualization.base
.. currentmodule:: autogluon.eda.visualization.base

.. autosummary::
   :nosignatures:

   AbstractVisualization

:hidden:`AbstractVisualization`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AbstractVisualization
   :members:
   :inherited-members:
