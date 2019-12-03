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

AutoGluon Applications
----------------------

.. autosummary::
   :nosignatures:

   ImageClassification
   ObjectDetection
   TabularPrediction

:hidden:`ImageClassification`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ImageClassification
   :members:
   :inherited-members:

    .. rubric:: Methods

    .. autoautosummary:: ImageClassification
        :methods:

:hidden:`ObjectDetection`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ObjectDetection
   :members:
   :inherited-members:

    .. rubric:: Methods

    .. autoautosummary:: ObjectDetection
        :methods:


.. automodule:: autogluon.task.image_classification

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

:hidden:`TabularPrediction`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TabularPrediction
   :members:
   :inherited-members:

    .. rubric:: Methods

    .. autoautosummary:: TabularPrediction
        :methods:

.. automodule:: autogluon.task.tabular_prediction
