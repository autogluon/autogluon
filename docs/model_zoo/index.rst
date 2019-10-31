NAS Model-Zoo
=============

Pretrained Models discovered via Neural Architecture Search

How To Use Pretrained Models
----------------------------


How to load pretrained 'efficientnet_b0'.

.. code-block:: python

   import autogluon as ag
   model = ag.nas.get_model('efficientnet_b0', pretrained=True)


EfficientNet
------------

The pretrained EfficientNet [1]_ models are provided.

.. [1] Tan, Mingxing, and Quoc V. Le. \
       "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import math
   import autogluon as ag
   from autogluon import ImageClassification as task

   @ag.autogluon_object(
       width_coefficient=ag.space.Categorical(1.1, 1.2),
       depth_coefficient=ag.space.Categorical(1.1, 1.2),
   )
   class EfficientNetB1(ag.nas.EfficientNet):
       def __init__(self, width_coefficient, depth_coefficient):
           input_factor = 2.0 / width_coefficient / depth_coefficient
           input_size = math.ceil((224 * input_factor) / 32) * 32
           super().__init__(width_coefficient=width_coefficient,
                            depth_coefficient=depth_coefficient,
                            input_size=input_size)

   task.fit('imagenet', net=EfficientNetB1(), search_strategy='grid',
            optimizer=ag.optimizer.SGD(learning_rate=1e-1, momentum=0.9, wd=1e-4))

   ag.done()
