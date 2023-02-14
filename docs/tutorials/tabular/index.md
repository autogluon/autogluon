# Tabular

For standard datasets that are represented as tables (stored as CSV file, parquet from database, etc.), AutoGluon can produce models to predict the values in one column based on the values in the other columns. With just a single call to `fit()`, you can achieve high accuracy in standard supervised learning tasks (both classification and regression), without dealing with cumbersome issues like data cleaning, feature engineering, hyperparameter optimization, model selection, etc.

::::{grid} 2
  :gutter: 3

:::{grid-item-card} Quick Start
  :link: tabular-quick-start.html

  5 min tutorial on fitting models with tabular datasets.
:::

:::{grid-item-card} Essentials
  :link: tabular-essentials.html

  Essential information about the most important settings for tabular prediction.
:::

:::{grid-item-card} In-depth
  :link: tabular-indepth.html

  In-depth tutorial on controlling various aspects of model fitting.
:::

:::{grid-item-card} Data Tables Containing Image, Text, and Tabular
  :link: tabular-multimodal.html

  Modeling data tables with image, text, numeric, and categorical features.
:::

:::{grid-item-card} Feature Engineering
  :link: tabular-feature-engineering.html

  AutoGluon's default feature engineering and how to extend it.
:::

:::{grid-item-card} Multi-Label Prediction
  :link: advanced/tabular-multilabel.html

  How to predict multiple columns in a data table.
:::

:::{grid-item-card} Kaggle Tutorial
  :link: advanced/tabular-kaggle.html

  Using AutoGluon for Kaggle competitions with tabular data.
:::

:::{grid-item-card} Training models with GPU support
  :link: advanced/tabular-gpu.html

  How to train models with GPU support.
:::

:::{grid-item-card} Interpretable rule-based modeling
  :link: advanced/tabular-interpretability.html

  Fitting interpretable models to data table for understanding data and predictions.
:::

:::{grid-item-card} Adding a Custom Metric
  :link: advanced/tabular-custom-metric.html

  How to add a custom metric to AutoGluon.
:::

:::{grid-item-card} Adding a Custom Model
  :link: advanced/tabular-custom-model.html

  How to add a custom model to AutoGluon.
:::

:::{grid-item-card} Adding a Custom Model (Advanced)
  :link: advanced/tabular-custom-model-advanced.html

  How to add a custom model to AutoGluon (Advanced).
:::

:::{grid-item-card} Deployment Optimization
  :link: advanced/tabular-deployment.html

  Tutorial on optimizing the predictor artifact for production deployment.
:::

::::

```{toctree}
---
maxdepth: 2
hidden: true
---

Essentials <tabular-essentials>
In Depth <tabular-indepth>
Feature Engineering <tabular-feature-engineering>
Tabular + Text + Images <tabular-multimodal>
Advanced <advanced/index>
```
