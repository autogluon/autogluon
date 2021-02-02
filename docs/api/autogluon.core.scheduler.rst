.. role:: hidden
    :class: hidden-section

autogluon.core.scheduler
=========================

.. admonition:: Example

   Define a toy training function with searchable spaces:

   >>> import numpy as np
   >>> import autogluon.core as ag
   >>> @ag.args(
   ...     lr=ag.space.Real(1e-3, 1e-2, log=True),
   ...     wd=ag.space.Real(1e-3, 1e-2),
   ...     epochs=10)
   >>> def train_fn(args, reporter):
   ...     print('lr: {}, wd: {}'.format(args.lr, args.wd))
   ...     for e in range(args.epochs):
   ...         dummy_accuracy = 1 - np.power(1.8, -np.random.uniform(e, 2*e))
   ...         reporter(epoch=e+1, accuracy=dummy_accuracy, lr=args.lr, wd=args.wd)

   Note that `epoch` returned by `reporter` must be the number of epochs done,
   and start with 1. Create a scheduler and use it to run training jobs:

   >>> scheduler = ag.scheduler.HyperbandScheduler(
   ...     train_fn,
   ...     resource={'num_cpus': 2, 'num_gpus': 0},
   ...     num_trials=100,
   ...     reward_attr='accuracy',
   ...     time_attr='epoch',
   ...     grace_period=1,
   ...     reduction_factor=3,
   ...     type='stopping')
   >>> scheduler.run()
   >>> scheduler.join_jobs()

   Note that `HyperbandScheduler` obtains the maximum number of epochs from
   `train_fn.args.epochs` (specified by `epochs=10` in the example above in the
   ag.args decorator). The value can also be passed as `max_t`
   to `HyperbandScheduler`.

   Visualize the results:

   >>> scheduler.get_training_curves(plot=True)

   .. image:: https://raw.githubusercontent.com/zhanghang1989/AutoGluonWebdata/master/doc/api/autogluon.1.png
      :width: 400

Schedulers
----------

.. currentmodule:: autogluon.core.scheduler

.. autosummary::
   :nosignatures:

   FIFOScheduler
   HyperbandScheduler
   RLScheduler

:hidden:`FIFOScheduler`
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: FIFOScheduler
   :members:
   :inherited-members:

    .. rubric:: Methods

    .. autoautosummary:: FIFOScheduler
        :methods:

:hidden:`HyperbandScheduler`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: HyperbandScheduler
   :members:
   :inherited-members:

    .. rubric:: Methods

    .. autoautosummary:: HyperbandScheduler
        :methods:


:hidden:`RLScheduler`
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RLScheduler
   :members:
   :inherited-members:

    .. rubric:: Methods

    .. autoautosummary:: RLScheduler
        :methods:
