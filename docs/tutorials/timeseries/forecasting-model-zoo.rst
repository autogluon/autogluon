.. _forecasting_zoo:

Forecasting Time Series - Model Zoo
===================================


.. note::

   This documentation is intended for advanced users and may not be comprehensive.

   For a stable public API, refer to TimeSeriesPredictor.


This page contains the list of time series forecasting models available in AutoGluon.
The available hyperparameters for each model are listed under **Other Parameters**.

This list is useful if you want to override the default hyperparameters (:ref:`sec_forecasting_indepth_manual_config`)
or define custom hyperparameter search spaces (:ref:`sec_forecasting_indepth_hpo`), as described in the In-depth Tutorial.
For example, the following code will train a ``TimeSeriesPredictor`` with ``DeepAR`` and ``ETS`` models with default hyperparameters (and a weighted ensemble on top of them)::

   predictor = TimeSeriesPredictor().fit(
      train_data,
      hyperparameters={
         "DeepAR": {},
         "ETS": {},
      },
   )

Note that we don't include the ``Model`` suffix when specifying the model name in ``hyperparameters``
(e.g., the class :class:`~autogluon.timeseries.models.DeepARModel` corresponds to the name ``"DeepAR"`` in the ``hyperparameters`` dictionary).


Also note that some of the models' hyperparameters have names and default values that
are different from the original libraries.

Default models
--------------

.. automodule:: autogluon.timeseries.models
.. currentmodule:: autogluon.timeseries.models

.. autosummary::
   :nosignatures:

   NaiveModel
   SeasonalNaiveModel
   ARIMAModel
   ETSModel
   ThetaModel
   AutoETSModel
   AutoARIMAModel
   DynamicOptimizedThetaModel
   AutoGluonTabularModel
   DeepARModel
   SimpleFeedForwardModel


:hidden:`NaiveModel`
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: NaiveModel
   :members: init

:hidden:`SeasonalNaiveModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SeasonalNaiveModel
   :members: init


:hidden:`ARIMAModel`
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ARIMAModel
   :members: init


:hidden:`ETSModel`
~~~~~~~~~~~~~~~~~~

.. autoclass:: ETSModel
   :members: init


:hidden:`ThetaModel`
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ThetaModel
   :members: init

:hidden:`AutoETSModel`
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AutoETSModel
   :members: init

:hidden:`AutoARIMAModel`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AutoARIMAModel
   :members: init

:hidden:`DynamicOptimizedThetaModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: DynamicOptimizedThetaModel
   :members: init

:hidden:`AutoGluonTabularModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AutoGluonTabularModel
   :members: init


:hidden:`DeepARModel`
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: DeepARModel
   :members: init


:hidden:`SimpleFeedForwardModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SimpleFeedForwardModel
   :members: init



MXNet Models
------------
Following MXNet-based models from GluonTS are available in AutoGluon.

- ``DeepARMXNetModel``
- ``MQCNNMXNetModel``
- ``MQRNNMXNetModel``
- ``SimpleFeedForwardMXNetModel``
- ``TemporalFusionTransformerMXNetModel``
- ``TransformerMXNetModel``

Documentation and hyperparameter settings for these models can be found `here <https://github.com/autogluon/autogluon/blob/master/timeseries/src/autogluon/timeseries/models/gluonts/mx/models.py>`_.

Using the above models requires installing Apache MXNet v1.9. This can be done as follows::

   python -m pip install mxnet~=1.9

If you want to use a GPU, install the version of MXNet that matches your CUDA version. See the
MXNet `documentation <https://mxnet.apache.org/versions/1.9.1/get_started?>`_ for more info.

If a GPU is available and MXNet version with CUDA is installed, all the MXNet models will be trained using the GPU.
Otherwise, the models will be trained on CPU.


Additional features
-------------------
Overview of the additional features and covariates supported by different models.
Models not included in this table currently do not support any additional features.

.. list-table::
   :header-rows: 1
   :stub-columns: 1
   :align: center
   :widths: 40 15 15 15 15

   * - Model
     - Static features (continuous)
     - Static features (categorical)
     - Known covariates (continuous)
     - Past covariates (continuous)
   * - :class:`~autogluon.timeseries.models.AutoGluonTabularModel`
     - ✓
     - ✓
     - ✓
     - ✓
   * - :class:`~autogluon.timeseries.models.DeepARModel`
     - ✓
     - ✓
     - ✓
     -
   * - :class:`~autogluon.timeseries.models.gluonts.mx.DeepARMXNetModel`
     - ✓
     - ✓
     - ✓
     -
   * - :class:`~autogluon.timeseries.models.gluonts.mx.MQCNNMXNetModel`
     - ✓
     - ✓
     - ✓
     - ✓
   * - :class:`~autogluon.timeseries.models.gluonts.mx.TemporalFusionTransformerMXNetModel`
     - ✓
     -
     - ✓
     -
