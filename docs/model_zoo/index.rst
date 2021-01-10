autogluon.model_zoo
===================

Here we provide pretrained Models discovered via Neural Architecture Search

How To Use Pretrained Models
----------------------------


Example showing how to load pretrained network 'efficientnet_b0', which was produced via NAS.

.. code-block:: python

   import autogluon.core as ag
   model = ag.model_zoo.get_model('efficientnet_b0', pretrained=True)


EfficientNet
------------

The following pretrained EfficientNet [1]_ models are provided for image classification.
The accuracy achieved by each model on a popular image classification benchmark is indicated, along with the image crop-size used by each model.

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


How to reproduce EfficientNet's neural architecture search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import math
   import autogluon.core as ag
   from autogluon.vision import ImagePredictor as Task

   @ag.obj(
       width_coefficient=ag.space.Categorical(1.1, 1.2),
       depth_coefficient=ag.space.Categorical(1.1, 1.2),
   )
   class EfficientNetB1(ag.model_zoo.EfficientNet):
       def __init__(self, width_coefficient, depth_coefficient):
           input_factor = 2.0 / width_coefficient / depth_coefficient
           input_size = math.ceil((224 * input_factor) / 32) * 32
           super().__init__(width_coefficient=width_coefficient,
                            depth_coefficient=depth_coefficient,
                            input_size=input_size)

   task = Task()
   task.fit('imagenet', search_strategy='grid',
            hyperparameters={'net': EfficientNetB1(), 'optimizer':ag.optimizer.SGD(learning_rate=1e-1, momentum=0.9, wd=1e-4)})
