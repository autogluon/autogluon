.. role:: hidden
    :class: hidden-section

autogluon.task
===============

.. admonition:: Example (image classification task):
   
   Tell AutoGluon that task is image classification:
   
   >>> import autogluon as ag
   >>> from autogluon import ImageClassification as task
   
   Load a toy image dataset:
   
   >>> filename = ag.download('http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/data.zip')
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

.. autosummary::
   :nosignatures:

   TabularPrediction
   ImageClassification
   ObjectDetection
   TextClassification


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


:hidden:`TextClassification`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TextClassification
   :members:
   :inherited-members:
   :exclude-members: run_fit

    .. rubric:: Methods

    .. autoautosummary:: TextClassification
        :methods:


Extra Methods/Classes for Tabular Prediction
--------------------------------------------

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


Extra Methods/Classes for Image Classification
----------------------------------------------

.. automodule:: autogluon.task.image_classification

:hidden:`Classifier`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Classifier
   :members: predict, evaluate, load, save, predict_proba
   :no-inherited-members:

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
   :inherited-members:

    .. rubric:: Methods

    .. autoautosummary:: ImageFolderDataset
        :methods:

    .. rubric:: Attributes

    .. autoautosummary:: ImageFolderDataset
        :attributes:

:hidden:`RecordDataset`
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RecordDataset
   :members:
   :inherited-members:

    .. rubric:: Methods

    .. autoautosummary:: RecordDataset
        :methods:

    .. rubric:: Attributes

    .. autoautosummary:: RecordDataset
        :attributes:


Extra Methods/Classes for Object Detection
------------------------------------------

.. automodule:: autogluon.task.object_detection

:hidden:`Detector`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Detector
   :members: predict, evaluate
   :no-inherited-members:

    .. rubric:: Methods

    .. autoautosummary:: Detector
        :methods:

    .. rubric:: Attributes

    .. autoautosummary:: Detector
        :attributes:

:hidden:`get_dataset`
~~~~~~~~~~~~~~~~~~~~~

.. autofunction::
   get_dataset

:hidden:`CustomVOCDetectionBase`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CustomVOCDetectionBase
   :members:
   :no-inherited-members:

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
   :no-inherited-members:

    .. rubric:: Methods

    .. autoautosummary:: COCO
        :methods:

    .. rubric:: Attributes

    .. autoautosummary:: COCO
        :attributes:


Extra Methods/Classes for Text Classification
----------------------------------------------

.. automodule:: autogluon.task.text_classification

:hidden:`TextClassificationPredictor`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TextClassificationPredictor
   :members: predict, evaluate, predict_proba
   :no-inherited-members:

    .. rubric:: Methods

    .. autoautosummary:: TextClassificationPredictor
        :methods:

    .. rubric:: Attributes

    .. autoautosummary:: TextClassificationPredictor
        :attributes:

:hidden:`get_dataset`
~~~~~~~~~~~~~~~~~~~~~

.. autofunction::
   get_dataset
