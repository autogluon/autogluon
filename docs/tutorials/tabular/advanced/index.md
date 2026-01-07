# Advanced Tabular Tutorials

For standard datasets that are represented as tables (stored as CSV file, parquet from database, etc.), AutoGluon can produce models to predict the values in one column based on the values in the other columns. With just a single call to `fit()`, you can achieve high accuracy in standard supervised learning tasks (both classification and regression), without dealing with cumbersome issues like data cleaning, feature engineering, hyperparameter optimization, model selection, etc.

::::{grid} 2
  :gutter: 3

:::{grid-item-card} Multi-Label Prediction
  :link: tabular-multilabel.html

  How to predict multiple columns in a data table.
:::

:::{grid-item-card} Kaggle Tutorial
  :link: tabular-kaggle.html

  Using AutoGluon for Kaggle competitions with tabular data.
:::

:::{grid-item-card} Training models with GPU support
  :link: tabular-gpu.html

  How to train models with GPU support.
:::

:::{grid-item-card} Adding a Custom Metric
  :link: tabular-custom-metric.html

  How to add a custom metric to AutoGluon.
:::

:::{grid-item-card} Adding a Custom Model
  :link: tabular-custom-model.html

  How to add a custom model to AutoGluon.
:::

:::{grid-item-card} Adding a Custom Model (Advanced)
  :link: tabular-custom-model-advanced.html

  How to add a custom model to AutoGluon (Advanced).
:::

:::{grid-item-card} Deployment Optimization
  :link: tabular-deployment.html

  Tutorial on optimizing the predictor artifact for production deployment.
:::

:::{grid-item-card} Hyperparameter Optimization
  :link: tabular-hpo.html

  Use hyperparameter optimization in AutoGluon.
:::
::::

```{toctree}
---
maxdepth: 1
hidden: true
---

Multilabel <tabular-multilabel>
Kaggle <tabular-kaggle>
GPU <tabular-gpu>
Custom Metrics <tabular-custom-metric>
Custom Models <tabular-custom-model>
Custom Models Advanced <tabular-custom-model-advanced>
Deployment <tabular-deployment>
Hyperparameter Optimization <tabular-hpo>
```
