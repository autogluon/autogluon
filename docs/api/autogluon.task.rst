.. role:: hidden
    :class: hidden-section

autogluon.task
===============

.. admonition:: Example (image classification task):
   
   Tell AutoGluon that task is image classification:
   
   >>> import autogluon as ag
   >>> from autogluon import ImageClassification as task
   
   Load a toy image dataset:
   
   >>> filename = ag.download('http://autogluon-hackathon.s3.amazonaws.com/data.zip')
   >>> ag.unzip(filename)
   >>> dataset = task.Dataset(train_path='data/train')
   
   Fit classification models:
   
   >>> classifier = task.fit(dataset, epochs=2)
   
   Evaluate predictions on test data:
   
   >>> test_dataset = task.Dataset('data/test', train=False)
   >>> test_acc = classifier.evaluate(test_dataset)


.. automodule:: autogluon.task

AutoGluon Tasks
-----------------

Prediction tasks built into AutoGluon such that a single call to `fit()` can produce high-quality trained models. For other applications, you can still use AutoGluon to tune the hyperparameters of your own custom models and training scripts.

.. autosummary::
   :nosignatures:

   TabularPrediction
   ImageClassification
   ObjectDetection
   TextPrediction


:hidden:`TabularPrediction`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TabularPrediction
   :members: fit, load, Predictor, Dataset
   :no-inherited-members:
   :exclude-members: run_fit

    .. rubric:: Methods

    .. autoautosummary:: TabularPrediction
        :methods:


:hidden:`ImageClassification`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ImageClassification
   :members:
   :inherited-members:
   :exclude-members: run_fit

    .. rubric:: Methods

    .. autoautosummary:: ImageClassification
        :methods:


:hidden:`ObjectDetection`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ObjectDetection
   :members:
   :inherited-members:
   :exclude-members: run_fit

    .. rubric:: Methods

    .. autoautosummary:: ObjectDetection
        :methods:


:hidden:`TextPrediction`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TextPrediction
   :members:
   :inherited-members:
   :exclude-members: run_fit

    .. rubric:: Methods

    .. autoautosummary:: TextPrediction
        :methods:


Additional Tabular Prediction APIs
----------------------------------

.. automodule:: autogluon.task.tabular_prediction

:hidden:`TabularPredictor`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TabularPredictor
   :members:
   :no-inherited-members:

    .. rubric:: Methods

    .. autoautosummary:: TabularPredictor
        :methods:

    .. rubric:: Attributes

    .. autoautosummary:: TabularPredictor
        :attributes:

:hidden:`TabularDataset`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TabularDataset
   :no-inherited-members:


Additional Image Classification APIs
------------------------------------

.. automodule:: autogluon.task.image_classification

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

.. automodule:: autogluon.task.object_detection

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

.. automodule:: autogluon.task.object_detection.dataset

:hidden:`get_dataset`
~~~~~~~~~~~~~~~~~~~~~

.. autofunction::
   get_dataset

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

.. automodule:: autogluon.task.text_prediction.models.basic_v1

:hidden:`BertForTextPredictionBasic`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: BertForTextPredictionBasic
   :members: predict, predict_proba, evaluate, save, load

    .. rubric:: Methods

    .. autoautosummary:: BertForTextPredictionBasic
        :methods:

    .. rubric:: Attributes

    .. autoautosummary:: BertForTextPredictionBasic
        :attributes:

