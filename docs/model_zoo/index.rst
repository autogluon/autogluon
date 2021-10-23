autogluon.extra.model_zoo
=========================

Here we provide pretrained Models discovered via Neural Architecture Search

How To Use Pretrained Models
----------------------------


Example showing how to load pretrained network 'efficientnet_b0', which was produced via NAS.

.. code-block:: python

   from autogluon.extra import model_zoo
   model = model_zoo.get_model('efficientnet_b0', pretrained=True)

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
