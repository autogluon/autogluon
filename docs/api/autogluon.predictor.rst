.. role:: hidden
    :class: hidden-section

AutoGluon Predictors
====================

.. admonition:: Example (Predictor for tabular data):

   Import TabularDataset and TabularPredictor:

   >>> from autogluon.tabular import TabularDataset, TabularPredictor

   Load a tabular dataset:

   >>> train_data = TabularDataset("https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv")

   Fit classification models predicting the "class" column:

   >>> predictor = TabularPredictor(label="class").fit(train_data)

   Load test data:

   >>> test_data = TabularDataset("https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv")

   Evaluate predictions on test data:

   >>> leaderboard = predictor.leaderboard(test_data)

.. admonition:: Example (Deep learning predictor for image, text and multimodal data):

   Import MultiModalPredictor:

   >>> from autogluon.multimodal import MultiModalPredictor
   >>> from datasets import load_dataset

   Load a multimodal data table:

   >>> train_data = load_dataset("glue", 'mrpc')['train'].to_pandas().drop('idx', axis=1)

   Fit classification models predicting the "class" column:

   >>> predictor = MultiModalPredictor(label="label").fit(train_data)

   Load test data:

   >>> test_data = load_dataset("glue", 'mrpc')['validation'].to_pandas().drop('idx', axis=1)

   Evaluate predictions on test data:

   >>> score = predictor.evaluate(test_data)


Predictors
----------

Predictors built into AutoGluon such that a single call to `fit()` can produce high-quality trained models for tabular, image, or text data. For other applications, you can still use AutoGluon to tune the hyperparameters of your own custom models and training scripts.

.. automodule:: autogluon.tabular
.. autosummary::
   :nosignatures:

   TabularPredictor

.. automodule:: autogluon.multimodal
.. autosummary::
   :nosignatures:

   MultiModalPredictor

.. automodule:: autogluon.timeseries
.. autosummary::
   :nosignatures:

   TimeSeriesPredictor


:hidden:`TabularPredictor`
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: autogluon.tabular

.. autoclass:: TabularPredictor
   :members:
   :inherited-members:
   :exclude-members: from_learner

    .. rubric:: Methods

    .. autoautosummary:: TabularPredictor
        :methods:


:hidden:`MultiModalPredictor`
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: autogluon.multimodal

.. autoclass:: MultiModalPredictor
   :members:
   :inherited-members:

    .. rubric:: Methods

    .. autoautosummary:: MultiModalPredictor
        :methods:


:hidden:`TimeSeriesPredictor`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: autogluon.timeseries

.. autoclass:: TimeSeriesPredictor
   :members:
   :inherited-members:
   :exclude-members: refit_full, score

    .. rubric:: Methods

    .. autoautosummary:: TimeSeriesPredictor
        :methods:


Additional Tabular APIs
-----------------------

.. automodule:: autogluon.tabular

:hidden:`TabularDataset`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TabularDataset
   :members: init

.. automodule:: autogluon.common.features.feature_metadata

:hidden:`FeatureMetadata`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: FeatureMetadata
   :members:
   :inherited-members:


Additional Time Series APIs
---------------------------

.. automodule:: autogluon.timeseries

:hidden:`TimeSeriesDataFrame`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TimeSeriesDataFrame
    :members: freq, fill_missing_values, from_iterable_dataset, from_data_frame, from_pickle,
              split_by_time, slice_by_timestep, slice_by_time, to_regular_index
