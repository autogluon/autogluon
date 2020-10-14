.. role:: hidden
    :class: hidden-section

AutoGluon Tasks
===============

.. admonition:: Example (image classification task):

   Tell AutoGluon that task is image classification:

   >>> import autogluon.core as ag
   >>> from autogluon.vision import ImageClassification as task

   Load a toy image dataset:

   >>> filename = ag.download('http://autogluon-hackathon.s3.amazonaws.com/data.zip')
   >>> ag.unzip(filename)
   >>> dataset = task.Dataset(train_path='data/train')

   Fit classification models:

   >>> classifier = task.fit(dataset, epochs=2)

   Evaluate predictions on test data:

   >>> test_dataset = task.Dataset('data/test', train=False)
   >>> test_acc = classifier.evaluate(test_dataset)



Tasks
-----

Prediction tasks built into AutoGluon such that a single call to `fit()` can produce high-quality trained models. For other applications, you can still use AutoGluon to tune the hyperparameters of your own custom models and training scripts.

.. automodule:: autogluon.tabular
.. autosummary::
   :nosignatures:

   TabularPrediction

.. automodule:: autogluon.vision
.. autosummary::
   :nosignatures:

   ImageClassification
   ObjectDetection

.. automodule:: autogluon.text
.. autosummary::
   :nosignatures:

   TextPrediction


:hidden:`TabularPrediction`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: autogluon.tabular

.. autoclass:: TabularPrediction
   :members: fit, load, Predictor, Dataset
   :no-inherited-members:
   :exclude-members: run_fit

    .. rubric:: Methods

    .. autoautosummary:: TabularPrediction
        :methods:


:hidden:`ImageClassification`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: autogluon.vision

.. autoclass:: ImageClassification
   :members:
   :inherited-members:
   :exclude-members: run_fit

    .. rubric:: Methods

    .. autoautosummary:: ImageClassification
        :methods:


:hidden:`ObjectDetection`
~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: autogluon.vision

.. autoclass:: ObjectDetection
   :members:
   :inherited-members:
   :exclude-members: run_fit

    .. rubric:: Methods

    .. autoautosummary:: ObjectDetection
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


Additional Tabular Prediction APIs
----------------------------------

.. automodule:: autogluon.tabular.tabular_prediction

:hidden:`TabularPredictor`
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: autogluon.tabular

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

