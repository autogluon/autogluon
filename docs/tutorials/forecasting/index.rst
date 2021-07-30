Temporary placeholder file
=======
Time-Series Forecasting
========================

For time-series data containing multiple individual series, AutoGluon can produce forecasting models to predict future values of each series based on historical observations of both this series and all of the other series in the dataset. A single call to AutoGluon's `fit()` automatically trains multiple models over a dataset containing multiple time-series measured over the same time period, and does not require you to manually deal with cumbersome issues like data cleaning, hyperparameter optimization, model selection, etc. Most models are from the `GluonTS library <https://ts.gluon.ai/>`_. Allowed to contain missing values and additional (non-time-varying) static features, the data can be loaded from: CSV file, parquet from database, or `GluonTS format <https://ts.gluon.ai/api/gluonts/gluonts.dataset.html>`_.


.. container:: cards

   .. card::
      :title: Quick Start
      :link: forecasting-quickstart.html

      5 min tutorial on fitting models with time-series datasets.

   .. card::
      :title: In-depth Tutorial
      :link: forecasting-indepth.html

      More advanced tutorial on controlling various aspects of time-series modeling.

   .. card::
      :title: FAQ
      :link: forecasting-faq.html

      Frequently asked questions about AutoGluon-Forecasting.

.. toctree::
   :maxdepth: 1
   :hidden:

   forecasting-quickstart
   forecasting-indepth
   forecasting-faq
