.. role:: hidden
    :class: hidden-section

AutoGluon Tasks
===============

.. note::

   TabularPredictor is replacing TabularPrediction in the upcoming v0.1 release. TabularPredictor has a very similar API to TabularPrediction, but certain arguments have been renamed. Please refer to the TabularPredictor documentation below for details, and refer to the updated tutorials which use TabularPredictor.


.. admonition:: Example (Tabular Prediction Task):

   Import TabularDataset and TabularPredictor:

   >>> from autogluon.tabular import TabularDataset, TabularPredictor

   Load a tabular dataset:

   >>> train_data = TabularDataset("https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv")

   Fit classification models predicting the "class" column:

   >>> predictor = TabularPredictor(label="class").fit(train_data)

   Load test data:

   >>> test_data = TabularDataset("https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv")

   Evaluate predictions on test data:

   >>> test_acc = predictor.evaluate(test_data)



Tasks
-----

Prediction tasks built into AutoGluon such that a single call to `fit()` can produce high-quality trained models. For other applications, you can still use AutoGluon to tune the hyperparameters of your own custom models and training scripts.

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

   TextPrediction


:hidden:`TabularPredictor`
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: autogluon.tabular

.. autoclass:: TabularPredictor
   :members:
   :inherited-members:

    .. rubric:: Methods

    .. autoautosummary:: TabularPredictor
        :methods:


:hidden:`ImagePrediction`
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


:hidden:`TextPrediction`
~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: autogluon.text

.. autoclass:: TextPrediction
   :members:
   :inherited-members:
   :exclude-members: run_fit

    .. rubric:: Methods

    .. autoautosummary:: TextPrediction
        :methods:


Additional Tabular APIs
-----------------------

.. automodule:: autogluon.tabular

:hidden:`TabularDataset`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TabularDataset
   :members: init

.. automodule:: autogluon.core.features.feature_metadata

:hidden:`FeatureMetadata`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: FeatureMetadata
   :members:
   :inherited-members:


Additional Image Prediction APIs
--------------------------------

.. automodule:: autogluon.mxnet

:hidden:`Classifier`
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Classifier
   :members: predict, evaluate, load, save

    .. rubric:: Methods

    .. autoautosummary:: Classifier
        :methods:

    .. rubric:: Attributes

    .. autoautosummary:: Classifier
        :attributes:

:hidden:`get_dataset`
~~~~~~~~~~~~~~~~~~~~~

.. autofunction::
   get_dataset

:hidden:`ImageFolderDataset`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ImageFolderDataset
   :members:
   :inherited-members: init

    .. rubric:: Methods

    .. autoautosummary:: ImageFolderDataset
        :methods: init

    .. rubric:: Attributes

    .. autoautosummary:: ImageFolderDataset
        :attributes:

:hidden:`RecordDataset`
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RecordDataset
   :members:
   :inherited-members: init

    .. rubric:: Methods

    .. autoautosummary:: RecordDataset
        :methods: init

    .. rubric:: Attributes

    .. autoautosummary:: RecordDataset
        :attributes:


Additional Text Prediction APIs
-------------------------------

.. automodule:: autogluon.text.text_prediction.models.basic_v1

:hidden:`BertForTextPredictionBasic`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: BertForTextPredictionBasic
   :members: predict, predict_proba, evaluate, save, load

    .. rubric:: Methods

    .. autoautosummary:: BertForTextPredictionBasic
        :methods:

    .. rubric:: Attributes

    .. autoautosummary:: BertForTextPredictionBasic
        :attributes:
