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



Predictors
----------

Predictors built into AutoGluon such that a single call to `fit()` can produce high-quality trained models for tabular, image, text, or time-series data. For other applications, you can still use AutoGluon to tune the hyperparameters of your own custom models and training scripts.

.. automodule:: autogluon.tabular
.. autosummary::
   :nosignatures:

   TabularPredictor

.. automodule:: autogluon.vision
.. autosummary::
   :nosignatures:

   ImagePredictor
   ObjectDetector

.. automodule:: autogluon.text
.. autosummary::
   :nosignatures:

   TextPredictor

.. automodule:: autogluon.forecasting
.. autosummary::
   :nosignatures:

   ForecastingPredictor


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


:hidden:`ImagePredictor`
~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: autogluon.vision

.. autoclass:: ImagePredictor
   :members:
   :inherited-members:
   :exclude-members: run_fit

    .. rubric:: Methods

    .. autoautosummary:: ImagePredictor
        :methods:


:hidden:`ObjectDetector`
~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: autogluon.vision

.. autoclass:: ObjectDetector
   :members:
   :inherited-members:
   :exclude-members: run_fit

    .. rubric:: Methods

    .. autoautosummary:: ObjectDetector
        :methods:


:hidden:`TextPredictor`
~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: autogluon.text

.. autoclass:: TextPredictor
   :members:
   :inherited-members:

    .. rubric:: Methods

    .. autoautosummary:: TextPredictor
        :methods:


:hidden:`ForecastingPredictor`
~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: autogluon.forecasting

.. autoclass:: ForecastingPredictor
   :members:
   :inherited-members:

    .. rubric:: Methods

    .. autoautosummary:: ForecastingPredictor
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

