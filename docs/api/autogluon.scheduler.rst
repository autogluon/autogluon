.. role:: hidden
    :class: hidden-section

autogluon.scheduler
===================

.. admonition:: Example

   Define a dummy training function with searchable spaces:

   >>> import numpy as np
   >>> import autogluon as ag
   >>> @ag.args(
   >>>     lr=ag.space.Real(1e-3, 1e-2, log=True),
   >>>     wd=ag.space.Real(1e-3, 1e-2))
   >>> def train_fn(args, reporter):
   ...     print('lr: {}, wd: {}'.format(args.lr, args.wd))
   ...     for e in range(10):
   ...         dummy_accuracy = 1 - np.power(1.8, -np.random.uniform(e, 2*e))
   ...         reporter(epoch=e, accuracy=dummy_accuracy, lr=args.lr, wd=args.wd)

   Create a schedule and run the experiment:

   >>> scheduler = ag.scheduler.HyperbandScheduler(train_fn,
   ...                                             resource={'num_cpus': 2, 'num_gpus': 0},
   ...                                             num_trials=10,
   ...                                             reward_attr='accuracy',
   ...                                             time_attr='epoch',
   ...                                             grace_period=1)
   >>> scheduler.run()
   >>> scheduler.join_jobs()

   Visiualize the results and exit:

   >>> scheduler.get_training_curves(plot=True)
   >>> ag.done()

   .. image:: https://raw.githubusercontent.com/zhanghang1989/AutoGluonWebdata/master/doc/api/autogluon.1.png
      :width: 400

Schedulers
----------

.. currentmodule:: autogluon.scheduler

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


Early Stopping Managers
-----------------------

.. autosummary::
   :nosignatures:

   HyperbandStopping_Manager
   HyperbandPromotion_Manager

:hidden:`HyperbandStopping_Manager`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: HyperbandStopping_Manager
   :members:
   :inherited-members:

    .. rubric:: Methods

    .. autoautosummary:: HyperbandStopping_Manager
        :methods:

:hidden:`HyperbandPromotion_Manager`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: HyperbandPromotion_Manager
   :members:
   :inherited-members:

    .. rubric:: Methods

    .. autoautosummary:: HyperbandPromotion_Manager
        :methods:
