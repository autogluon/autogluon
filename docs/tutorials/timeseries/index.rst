Time Series Forecasting
========================

For time-series data containing multiple individual series, AutoGluon can produce
forecasting models to predict future values of each series based on historical
observations of both this series and all of the other series in the dataset.
A single call to AutoGluon `TimeSeriesPredictor`'s `fit()` automatically trains
multiple models on a dataset containing multiple time-series measured over
the same time period, and does not require you to manually deal with cumbersome
issues like data cleaning, hyperparameter optimization, model selection, etc.

Most neural network-based models are from the `GluonTS library <https://ts.gluon.ai/>`_.
Allowed to contain missing values and additional (non-time-varying) static features,
the data can be loaded from: a CSV file
or the `GluonTS format <https://ts.gluon.ai/api/gluonts/gluonts.dataset.html>`_.
AutoGluon-TimeSeries also supports simpler time series models such as exponential
smoothing or ARIMA, through the `statsmodels library <https://www.statsmodels.org/stable/index.html>`_.



.. container:: cards

   .. card::
      :title: Quick Start
      :link: forecasting-quickstart.html

      Quick start tutorial on fitting models with time series datasets.

   .. card::
      :title: In-depth Tutorial
      :link: forecasting-indepth.html

      Detailed discussion of the time series forecasting capabilities in AutoGluon.

   .. card::
      :title: Model Zoo
      :link: forecasting-model-zoo.html

      List of available forecasting models in AutoGluon-TimeSeries.

   .. card::
      :title: FAQ
      :link: forecasting-faq.html

      Frequently asked questions about AutoGluon-TimeSeries.

.. toctree::
   :maxdepth: 1
   :hidden:

   forecasting-quickstart
   forecasting-indepth
   forecasting-model-zoo
   forecasting-faq
