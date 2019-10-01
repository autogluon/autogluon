Object Detection
---------------------

:download:`Download scripts </model_zoo/object_detection.zip>`


Object Detection on PASCAL VOC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ is a collection of
datasets for object detection. The most commonly combination for
benchmarking is using *2007 trainval* and *2012 trainval* for training and *2007
test* for validation.

Please follow `GluonCV data preparation <https://gluon-cv.mxnet.io/build/examples_datasets/pascal_voc.html>`_
to prepare the PASCAL VOC data.

.. editing URL for the following table: https://tinyurl.com/yym87n3p

+----------------+----------+-----------------+--------------+----------+----------+----------+
|                | Searcher | Trial scheduler | Val mAP (%)  | Time (h) | Command  | MXBoard  |
+================+==========+=================+==============+==========+==========+==========+
| Prev. SOTA [1] | -        | -               | 80.1         | -        | -        | -        |
+----------------+----------+-----------------+--------------+----------+----------+----------+
| AutoGluon      | Random   | FIFO            | 84.2         | 856.8    | Com. [1] | Fig. [1] |
+----------------+----------+-----------------+--------------+----------+----------+----------+
|                | Random   | Hyperband       | 79.3         | 599.0    | Com. [2] | Fig. [2] |
+----------------+----------+-----------------+--------------+----------+----------+----------+

Com. [1]:

.. code-block:: console

    $ python train_ssd.py --max_trial_count 200 --max_training_epochs 240

Fig. [1]:

    .. image:: ../img/voc_map_curves_1.svg

Com. [2]:

.. code-block:: console

    $ python train_ssd.py --max_trial_count 200 --max_training_epochs 240 --trial_scheduler hyperband

Fig. [2]:

    .. image:: ../img/voc_map_curves_2.svg

Reference
~~~~~~~~~~
[1] Liu, Wei, et al. "Ssd: Single shot multibox detector." European conference on computer vision. Springer, Cham, 2016.