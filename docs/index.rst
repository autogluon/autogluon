AutoGluon: AutoML Toolkit with MXNet Gluon
==========================================

`AutoGluon` enables easy-to-use and easy-to-extend AutoML with a focus on deep learning, and making AutoML deploy in real-world applications. Feature includes:

- Fast prototyping on deep learning applications.
- Automatic Hyper-parameter Optimization.
- Leveraging state-of-the-art deep learning techniques.
- Easy customization for advanced usage.
- Painless setup using distributed computations.

Installation
------------

PIP Install AutoGluon:

.. code-block:: bash

   pip install https://autogluon.s3.amazonaws.com/dist/autogluon-0.0.1-py3-none-any.whl

Please visit this `tutorial <http://mxnet.incubator.apache.org/get_started>`_ to install MXNet.

Tutorials
---------

This tutorial section is ...

.. container:: cards
   
   .. card::
      :title: Image Classification
      :link: tutorials/image_classification/index.html

      AutoGluon image classification tutorials.

   .. card::
      :title: Extending AutoGluon
      :link: tutorials/extend/index.html

      AutoGluon image classification tutorials.


.. toctree::
   :maxdepth: 1
   :hidden:

   tutorials/index
   api/index
   model_zoo/index

AutoGluon NAS Model Zoo
=======================

How To Use Pretrained Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

How to load pretrained 'efficientnet_b0'.

.. code-block:: python

   import autogluon as ag
   model = ag.get_model('efficientnet_b0', pretrained=True)


How to evaluate the model:

.. code-block:: python
      
   from autogluon import ImageClassification as task
   task.evaluate('imagenet', model='efficientnet_b0', input_size=224)


EfficientNet
~~~~~~~~~~~~

+---------------------------+--------+-----------+
| Model                     | Acc %  | crop_size |
+===========================+========+===========+
| EfficientNet_B0           | 77.03  | 224       |
+---------------------------+--------+-----------+
| EfficientNet_B1           | 78.66  | 240       |
+---------------------------+--------+-----------+
| EfficientNet_B2           | 79.57  | 260       |
+---------------------------+--------+-----------+
| EfficientNet_B3           | 80.68  | 300       |
+---------------------------+--------+-----------+
| EfficientNet_B4           | 81.97  | 380       |
+---------------------------+--------+-----------+
| EfficientNet_B5           | 83.30  | 456       |
+---------------------------+--------+-----------+
| EfficientNet_B6           | 83.79  | 528       |
+---------------------------+--------+-----------+
| EfficientNet_B7           | 83.86  | 600       |
+---------------------------+--------+-----------+


How to reproduce search on EfficientNet?

.. code-block:: python

   import math
   import autogluon as ag
   from autogluon import ImageClassification as task

   @ag.autogluon_object(
       width_coefficient=ag.Categorical(1.1, 1.2),
       depth_coefficient=ag.Categorical(1.1, 1.2),
   )
   class EfficientNetB1(ag.nas.EfficientNet):
       def __init__(self, width_coefficient, depth_coefficient):
           input_factor = 2.0 / width_coefficient / depth_coefficient
           input_size = math.ceil((224 * input_factor) / 32) * 32
           super().__init__(width_coefficient=width_coefficient,
                            depth_coefficient=depth_coefficient,
                            input_size=input_size)

   task.fit('imagenet', net=EfficientNetB1(), search_strategy='grid',
            optimizer=ag.optimizer.SGD(learning_rate=1e-1,momentum=0.9,wd=1e-4))

More Resources
--------------

- [Gluon Crush Course](https://beta.mxnet.io/guide/getting-started/crash-course/index.html)
- [MXNet Tutorials](https://mxnet.apache.org/api/python/docs/tutorials/).
- [MXNet API](https://mxnet.io/).
