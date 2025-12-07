# Time Series Forecasting

AutoGluon can forecast the future values of multiple time series given the historical data and other related covariates.
A single call to AutoGluon `TimeSeriesPredictor`'s `fit()` method trains multiple models to generate accurate probabilistic forecasts,
and does not require you to manually deal with cumbersome issues like model selection and hyperparameter tuning.

Under the hood, AutoGluon combines various state of the art forecasting algorithms.
These include established statical methods like ETS and ARIMA from
[`StatsForecast`](https://github.com/Nixtla/statsforecast),
efficient tree-based forecasters like LightGBM based on [AutoGluon-Tabular](https://auto.gluon.ai/stable/tutorials/tabular/index.html), flexible deep learning models like DeepAR and Temporal Fusion Transformer from [GluonTS](https://ts.gluon.ai/), and a pretrained zero-shot forecasting model, [Chronos](https://github.com/amazon-science/chronos-forecasting).

Check out the [Quick Start Tutorial](forecasting-quick-start.ipynb) to learn how to make accurate forecasts in just 3 lines of code using AutoGluon.

::::{grid} 2
  :gutter: 3

:::{grid-item-card} Quick Start
  :link: forecasting-quick-start.html

  Quick start tutorial on fitting models with time series datasets.
:::

:::{grid-item-card} In-depth Tutorial
  :link: forecasting-indepth.html

  Detailed discussion of the time series forecasting capabilities in AutoGluon.
:::

:::{grid-item-card} Forecasting with Chronos
  :link: forecasting-chronos.html

  Zero-shot forecasting with pretrained Chronos time series models in AutoGluon.
:::

:::{grid-item-card} Model Zoo
  :link: forecasting-model-zoo.html

  List of available forecasting models in AutoGluon-TimeSeries.
:::

:::{grid-item-card} Metrics
  :link: forecasting-metrics.html

  Evaluation metrics available in AutoGluon-TimeSeries.
:::

:::{grid-item-card} Custom Models
  :link: advanced/forecasting-custom-model.html

  How to add a custom time series forecasting model to AutoGluon.
:::

::::


```{toctree}
---
maxdepth: 2
hidden: true
---

Quick Start <forecasting-quick-start>
In Depth <forecasting-indepth>
Forecasting with Chronos <forecasting-chronos>
Metrics <forecasting-metrics>
Model Zoo <model_zoo/index>
Advanced <advanced/index>
```
