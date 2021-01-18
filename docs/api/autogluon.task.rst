.. role:: hidden
    :class: hidden-section

AutoGluon Tasks
===============

.. admonition:: Example (image prediction task):

   Tell AutoGluon that task is image prediction:

   >>> import autogluon.core as ag
   >>> from autogluon.vision import ImagePredictor as Task

   Load a toy image dataset:

   >>> train, val, test = Task.Dataset.from_folders('https://autogluon.s3.amazonaws.com/datasets/fashion-mnist-demo.zip')

   Fit classification models:

   >>> mytask = Task()
   >>> mytask.fit(train, epochs=2)

   Evaluate predictions on test data:

   >>> test_acc = mytask.evaluate(test)



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
~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: autogluon.text

.. autoclass:: TextPrediction
   :members:
   :inherited-members:
   :exclude-members: run_fit

    .. rubric:: Methods

    .. autoautosummary:: TextPrediction
        :methods:


Additional Tabular APIs
----------------------------------

.. automodule:: autogluon.tabular

:hidden:`TabularDataset`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TabularDataset
   :no-members:
   :no-inherited-members:

:hidden:`FeatureMetadata`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: FeatureMetadata
   :members:
   :inherited-members:


Additional Image Prediction APIs
--------------------------------

.. automodule:: autogluon.mxnet

:hidden:`Classifier`
~~~~~~~~~~~~~~~~~~~~~~~~~~

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


Additional Object Detection APIs
--------------------------------

.. automodule:: autogluon.vision

:hidden:`Detector`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Detector
   :members: predict, evaluate

    .. rubric:: Methods

    .. autoautosummary:: Detector
        :methods:

    .. rubric:: Attributes

    .. autoautosummary:: Detector
        :attributes:

.. automodule:: autogluon.vision.object_detection.dataset


:hidden:`CustomVOCDetection`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CustomVOCDetection
   :members:

    .. rubric:: Methods

    .. autoautosummary:: CustomVOCDetection
        :methods:

    .. rubric:: Attributes

    .. autoautosummary:: CustomVOCDetection
        :attributes:

:hidden:`CustomVOCDetectionBase`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CustomVOCDetectionBase
   :members:

    .. rubric:: Methods

    .. autoautosummary:: CustomVOCDetectionBase
        :methods:

    .. rubric:: Attributes

    .. autoautosummary:: CustomVOCDetectionBase
        :attributes:

:hidden:`COCO`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: COCO
   :members:

    .. rubric:: Methods

    .. autoautosummary:: COCO
        :methods:

    .. rubric:: Attributes

    .. autoautosummary:: COCO
        :attributes:


Additional Text Prediction APIs
-----------------------------------

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
