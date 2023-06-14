# Time Series Forecasting

AutoGluon can forecast the future values of multiple time series given the historical data and other related covariates.
A single call to AutoGluon `TimeSeriesPredictor`'s `fit()` method trains multiple models to generate accurate probabilistic forecasts,
and does not require you to manually deal with cumbersome issues like model selection and hyperparameter tuning.

Under the hood, AutoGluon combines various state of the art forecasting algorithms.
These include established statical methods like ETS and ARIMA from
[`StatsForecast`](https://github.com/Nixtla/statsforecast) and [`statsmodels`](https://statsmodels.org/) libraries,
efficient tree-based forecasters like LightGBM based on [AutoGluon-Tabular](https://auto.gluon.ai/stable/tutorials/tabular/index.html),
and flexible deep learning models like DeepAR and Temporal Fusion Transformer from [GluonTS](https://ts.gluon.ai/).

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

:::{grid-item-card} Model Zoo
  :link: forecasting-model-zoo.html

  List of available forecasting models in AutoGluon-TimeSeries.
:::

::::

```{toctree}
---
maxdepth: 1
hidden: true
---

Quick Start <forecasting-quick-start>
In Depth <forecasting-indepth>
Model Zoo <forecasting-model-zoo>
```
