.. role:: hidden
    :class: hidden-section

autogluon.task
===============

.. admonition:: Example
   
   Import AutoGluon and Image Classification task:
   
   >>> import autogluon as ag
   >>> from autogluon import ImageClassification as task
   
   Download a toy dataset:
   
   >>> filename = ag.download('http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/data.zip')
   >>> ag.unzip(filename)
   >>> dataset = task.Dataset(train_path='data/train')

   Start Fitting:

   >>> classifier = task.fit(dataset, epochs=2)

   Evaluate on Test Dataset:
   
   >>> test_dataset = task.Dataset(test_path='data/test')
   >>> test_acc = classifier.evaluate(test_dataset)
   >>> print('Top-1 test acc: %.3f' % test_acc)
   Top-1 test acc: 0.506

.. automodule:: autogluon.task

AutoGluon Applications
----------------------

.. autosummary::
   :nosignatures:

   ImageClassification

:hidden:`ImageClassification`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ImageClassification
   :members:
   :inherited-members:

    .. rubric:: Methods

    .. autoautosummary:: ImageClassification
        :methods:
