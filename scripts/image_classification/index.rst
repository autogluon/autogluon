Image Classification
---------------------

:download:`Download scripts </model_zoo/image_classification.zip>`


Image Classification on MINC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
`MINC <http://opensurfaces.cs.cornell.edu/publications/minc/>`__ is
short for Materials in Context Database, provided by Cornell.
``MINC-2500`` is a resized subset of ``MINC`` with 23 classes, and 2500
images in each class.

To start, we first download ``MINC-2500`` from
`here <http://opensurfaces.cs.cornell.edu/publications/minc/>`__.
Suppose we have the data downloaded to ``~/data/`` and
extracted to ``~/data/minc-2500``.

.. editing URL for the following table: https://tinyurl.com/yyt64zbk

+----------------------------+--------+----------+-----------------+---------------------------+----------+----------+----------+
|                            | Trials | Searcher | Trial scheduler | Test accuracy (%)         | Time (h) | Command  | MXBoard  |
+============================+========+==========+=================+===========================+==========+==========+==========+
| Transfer learning SOTA [1] | 1      | -        | -               | 77.6(finetune)/81.3(sota) | -        | -        | -        |
+----------------------------+--------+----------+-----------------+---------------------------+----------+----------+----------+
| AutoGluon                  | 400    | Random   | FIFO            | 80.3                      | 48       | Com. [1] | Fig. [1] |
+----------------------------+--------+----------+-----------------+---------------------------+----------+----------+----------+
|                            | 400    | Random   | Hyperband       | 80.0                      | 48       | Com. [2] | Fig. [2] |
+----------------------------+--------+----------+-----------------+---------------------------+----------+----------+----------+

Com. [1]:

.. code-block:: console

    $ python prepare_minc.py --data ~/data/minc-2500 --split 1
    $ python train_minc.py --data ~/data/minc-2500/ --nets resnet50_v1b
        --num_trials 400 --num_training_epochs 100 --batch_size 32
        --lr_factor 0.1 --lr_step 20 --num_gpus 2 --savedir checkpoint1/exp1.ag
        --time_limits 172800

Fig. [1]:

    .. image:: ../img/minc_accuracy_curves_1.svg

Com. [2]:

.. code-block:: console

    $ python prepare_minc.py --data ~/data/minc-2500 --split 1
    $ python train_minc.py --data ~/data/minc-2500/ --nets resnet50_v1b --num_trials 400
        --num_training_epochs 100 --batch_size 32 --lr_factor 0.1 --lr_step 20 --num_gpus 2
        --trial_scheduler hyperband --savedir checkpoint1/exp1.ag --time_limits 172800

Fig. [2]:

    .. image:: ../img/minc_accuracy_curves_2.svg


Image Classification on Shoppe-IET
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Shopee-IET <https://www.kaggle.com/c/shopee-iet-machine-learning-competition/>`_ aims
to build a model that can predict the classification of the input images.
The training/test sets are images provided by Shopee, which are classified into 18 categories.

.. editing URL for the following table: https://tinyurl.com/y3fmn64g

+--------------------+--------+----------+-----------------+--------------------------+----------+----------+----------+
|                    | Trials | Searcher | Trial scheduler | Test accuracy (%)        | Time (h) | Command  | MXBoard  |
+====================+========+==========+=================+==========================+==========+==========+==========+
| Leadboard SOTA [2] | 1      | -        | -               | 86.7                     | -        | -        | -        |
+--------------------+--------+----------+-----------------+--------------------------+----------+----------+----------+
| AutoGluon          | 400    | Random   | FIFO            | 85.0                     | 200.9    | Com. [3] | Fig. [3] |
+--------------------+--------+----------+-----------------+--------------------------+----------+----------+----------+
|                    | 400    | Random   | Hyperband       | 85.2                     | 132.1    | Com. [4] | Fig. [4] |
+--------------------+--------+----------+-----------------+--------------------------+----------+----------+----------+

Com. [3]:

.. code-block:: console

    $ sh download_shopeeiet.sh
    $ python prepare_shopeeiet.py --data ~/data/shopeeiet/ --split 9
    $ python train_shopeeiet.py --data ~/data/shopeeiet/
        --nets resnet152_v1d --num_trials 400 --num_training_epochs 100
        --batch_size 32 --lr_factor 0.1 --lr_step 20 --num_gpus 2
        --savedir checkpoint2/exp1.ag --time_limits 172800

Fig. [3]:

    .. image:: ../img/shopee_accuracy_curves_1.svg

Com. [4]:

.. code-block:: console

    $ sh download_shopeeiet.sh
    $ python prepare_shopeeiet.py --data ~/data/shopeeiet/ --split 9
    $ python train_shopeeiet.py --data ~/data/shopeeiet/ --nets resnet152_v1d
        --num_trials 400 --num_training_epochs 100 --batch_size 32
        --lr_factor 0.1 --lr_step 20 --num_gpus 2 --trial_scheduler hyperband
        --savedir checkpoint2/exp1.ag --time_limits 172800

Fig. [4]:

    .. image:: ../img/shopee_accuracy_curves_2.svg


Image Classification on CIFAR10
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`__ is a
dataset of tiny (32x32) images with labels, collected by Alex Krizhevsky,
Vinod Nair, and Geoffrey Hinton. It is widely used as benchmark in
computer vision research.

.. editing URL for the following table: https://tinyurl.com/y64fq9m7

+--------------------------------+--------+----------+-----------------+--------------------------+----------+----------+----------+
|                                | Trials | Searcher | Trial scheduler | Test accuracy (%)        | Time (h) | Command  | MXBoard  |
+================================+========+==========+=================+==========================+==========+==========+==========+
| Training from scratch SOTA [3] | 1      | -        | -               | 96.3                     | -        | -        | -        |
+--------------------------------+--------+----------+-----------------+--------------------------+----------+----------+----------+
| AutoGluon                      | 200    | Random   | FIFO            | 99.18                    | 48       | Com. [5] | Fig. [5] |
+--------------------------------+--------+----------+-----------------+--------------------------+----------+----------+----------+
|                                | 200    | Random   | Hyperband       | 99.16                    | 48       | Com. [6] | Fig. [6] |
+--------------------------------+--------+----------+-----------------+--------------------------+----------+----------+----------+

Com. [5]:

.. code-block:: console

    $ python train_cifar10.py --nets CIFAR_ResNeXt29_16x64d --num_trials 200
        --num_training_epochs 300 --batch_size 32 --num_gpus 4
        --savedir checkpoint1/exp1.ag --time_limits 172800

Fig. [5]:

    .. image:: ../img/cifar_accuracy_curves_1.svg

Com. [6]:

.. code-block:: console

    $ python train_cifar10.py --nets CIFAR_ResNeXt29_16x64d --num_trials 200
        --num_training_epochs 300 --batch_size 32 --num_gpus 4
        --trial_scheduler hyperband --savedir checkpoint2/exp1.ag --time_limits 172800

Fig. [6]:

    .. image:: ../img/cifar_accuracy_curves_2.svg


Reference
~~~~~~~~~~
[1] Zhang, Hang, Jia Xue, and Kristin Dana. "Deep ten: Texture encoding network." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
[2] https://www.kaggle.com/c/shopee-iet-machine-learning-competition/leaderboard
[3] https://gluon-cv.mxnet.io/model_zoo/classification.html