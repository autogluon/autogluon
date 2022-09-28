Forecasting Time Series - Model Zoo
===================================

.. note::

   This documentation is intended for advanced users and may not be comprehensive.

   For a stable public API, refer to TimeSeriesPredictor.



Models
------

This page contains the list of time series forecasting models available in AutoGluon.
The available hyperparameters for each model are listed under **Other parameters**.

This list is useful if you want to `override the default hyperparameters <forecasting-indepth.html#manually-configuring-models>`_ or `define custom hyperparameter search spaces <forecasting-indepth.html#hyperparameter-tuning>`_, as described in the In-depth Tutorial.

Please note that some of the models' hyperparameters have names and default values that
are different from the original libraries.

.. automodule:: autogluon.timeseries.models
.. currentmodule:: autogluon.timeseries.models

.. autosummary::
   :nosignatures:

   ARIMAModel
   ETSModel
   DeepARModel
   SimpleFeedForwardModel
   MQCNNModel
   MQRNNModel
   TransformerModel
   TemporalFusionTransformerModel
   SktimeTBATSModel


:hidden:`ARIMAModel`
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ARIMAModel
   :members: init


:hidden:`ETSModel`
~~~~~~~~~~~~~~~~~~

.. autoclass:: ETSModel
   :members: init


:hidden:`DeepARModel`
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: DeepARModel
   :members: init


:hidden:`SimpleFeedForwardModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SimpleFeedForwardModel
   :members: init


:hidden:`MQCNNModel`
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MQCNNModel
   :members: init


:hidden:`MQRNNModel`
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MQRNNModel
   :members: init


:hidden:`TransformerModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TransformerModel
   :members: init


:hidden:`TemporalFusionTransformerModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TemporalFusionTransformerModel
   :members: init


:hidden:`SktimeTBATSModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SktimeTBATSModel
   :members: init
