```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# Reference: Base APIs

This section highlights the base APIs used by the EDA framework. The processing consists of the following parts:

1. Analysis graph construction - in this part a nested graph of analyses is constructed.

```python3
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
```

2\. **.fit()** call. This call will execute operations in the graph and produce a **state**. The state is a nested
dictionary without any prescribed structure. All components share the same namespace. If multiple components
are fitted with different parameters, they can be put into separate sub-spaces via **Namespace** component
that can be passed either for further processing
via next analysis or be rendered.

```python3
state = analysis.fit()
```

3\. Rendering: in this stage we construct components graph (a combination of layout components and visual components) and
then pass **State** generated previously as an input argument into **render()** call.

```python3
viz = SimpleVerticalLinearLayout(
    facets=[
        DatasetStatistics(headers=True),
        DatasetTypeMismatch(headers=True),
        MarkdownSectionComponent("### Feature Distance"),
        FeatureDistanceAnalysisVisualization(),
    ],
)
viz.render(state)
```

Please note: it is possible that the components may depend on each other's output; all the pre-requisites to **fit()**
the component must be checked in **can_handle()**. There are two ways the components can share the information:
1\) using **state**; 2) share values/shadow arguments (i.e., sample component modifies **train_data**, **test_data**
and **val_data** arguments in the scope of calling children's **fit()**.

## autogluon.eda.analysis.base

```{eval-rst}
.. automodule:: autogluon.eda.analysis.base
```

```{eval-rst}
.. currentmodule:: autogluon.eda.analysis.base
```

```{eval-rst}
.. autosummary::
   :nosignatures:

   AbstractAnalysis
   Namespace
```

### {hidden}`AbstractAnalysis`

```{eval-rst}
.. autoclass:: AbstractAnalysis
   :members:
   :inherited-members:
```

### {hidden}`Namespace`

```{eval-rst}
.. autoclass:: Namespace
   :members: init
```

## autogluon.eda.visualization.base

```{eval-rst}
.. automodule:: autogluon.eda.visualization.base
```

```{eval-rst}
.. currentmodule:: autogluon.eda.visualization.base
```

```{eval-rst}
.. autosummary::
   :nosignatures:

   AbstractVisualization
```

### {hidden}`AbstractVisualization`

```{eval-rst}
.. autoclass:: AbstractVisualization
   :members:
   :inherited-members:
```
