# Exploratory Data Analysis Tools

This section contains a high-level overview and showcases for exploratory data analysis tools.

::::{grid} 2
  :gutter: 3

:::{grid-item-card} Automated Dataset Overview
   :link: eda-auto-dataset-overview.html

   Get a high-level understanding of datasets including basic statistical information and feature information.
:::

:::{grid-item-card} Automated Target Variable Analysis
   :link: eda-auto-target-analysis.html

   Analyze and summarize the variable we are trying to predict and it's relationship with other variables.
:::

:::{grid-item-card} Quick Model Fit
   :link: eda-auto-quick-fit.html

   Fit a quick model to understand the relationships between the label and the other features in a dataset.
:::

:::{grid-item-card} Covariate Shift Detection
   :link: eda-auto-covariate-shift.html

   Identify situations where the distribution of features in a dataset changes between the training
   and testing phases, which can lead to biased model performance.
:::

:::{grid-item-card} Feature Interaction Charts
   :link: eda-auto-analyze-interaction.html

   Visualize 1/2/3-way relationships between features via chart.
   The tool automatically picks a chart type given the types of input features.
:::

:::{grid-item-card} Anomaly Detection
   :link: eda-auto-anomaly-detection.html

   Explore anomaly detection tools to identify unusual patterns in data and make informed decisions.
:::

::::

## Main API Reference

The section contains a reference of base constructs and composite components.

::::{grid} 2
  :gutter: 3

:::{grid-item-card} Auto: High-level Composite Components
   :link: references/autogluon.eda.auto.html

   Reference for high-level composite components.
:::

:::{grid-item-card} Base APIs
   :link: references/autogluon.eda.base-apis.html

   Components building blocks APIs.

::::

## Low-level components API reference

The section contains a reference for low-level components.

::::{grid} 2
  :gutter: 3

:::{grid-item-card} autogluon.eda.dataset
   :link: components/autogluon.eda.dataset.html

   Dataset-level APIs
:::

:::{grid-item-card} autogluon.eda.interaction
   :link: components/autogluon.eda.interaction.html

   Feature-level interactions APIs
:::

:::{grid-item-card} autogluon.eda.missing
   :link: components/autogluon.eda.missing.html

   Missing data APIs
:::

:::{grid-item-card} autogluon.eda.model
   :link: components/autogluon.eda.model.html

   Model level APIs
:::

:::{grid-item-card} autogluon.eda.shift
   :link: components/autogluon.eda.shift.html

   Distribution shift APIs
:::

:::{grid-item-card} autogluon.eda.transform
   :link: components/autogluon.eda.transform.html

   Transformations APIs
:::

::::

```{toctree}
:hidden: true
:maxdepth: 1

eda-auto-dataset-overview
eda-auto-target-analysis
eda-auto-quick-fit
eda-auto-covariate-shift
eda-auto-analyze-interaction
eda-auto-anomaly-detection
References <references/index>
Components <components/index>
```
