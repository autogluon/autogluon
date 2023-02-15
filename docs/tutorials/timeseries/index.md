# Time Series Forecasting

For time series data containing multiple individual series, AutoGluon can produce
forecasting models to predict future values of each series based on historical
observations of both this series and all of the other series in the dataset.
A single call to AutoGluon `TimeSeriesPredictor`'s `fit()` automatically trains
multiple models on a dataset containing multiple time-series measured over
the same time period, and does not require you to manually deal with cumbersome
issues like data cleaning, hyperparameter optimization, model selection, etc.

Most neural network-based models are from the [GluonTS library](https://ts.gluon.ai/).
Allowed to contain missing values and additional (non-time-varying) static features,
the data can be loaded from: a CSV file
or the [GluonTS format](https://ts.gluon.ai/stable/api/gluonts/gluonts.dataset.html).
AutoGluon-TimeSeries also supports simpler time series models such as exponential
smoothing or ARIMA, through the [statsmodels library](https://www.statsmodels.org/stable/index.html).



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
