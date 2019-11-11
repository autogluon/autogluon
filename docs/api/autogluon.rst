autogluon.core
--------------

Search Space
~~~~~~~~~~~~

Core features in AutoGluon, including search spaces and decorators for customized search configuration
and user-defined objects.

.. admonition:: Example

   Define a dummy training function with searchable spaces:

   >>> import numpy as np
   >>> import autogluon as ag
   >>> @ag.args(
   >>>     lr=ag.space.Real(1e-3, 1e-2, log=True),
   >>>     wd=ag.space.Real(1e-3, 1e-2))
   >>> def train_fn(args, reporter):
   >>>     print('lr: {}, wd: {}'.format(args.lr, args.wd))
   >>>     for e in range(10):
   >>>         dummy_accuracy = 1 - np.power(1.8, -np.random.uniform(e, 2*e))
   >>>         reporter(epoch=e, accuracy=dummy_accuracy, lr=args.lr, wd=args.wd)

   Create a schedule and run the experiment:

   >>> scheduler = ag.scheduler.HyperbandScheduler(train_fn,
   >>>                                             resource={'num_cpus': 2, 'num_gpus': 0},
   >>>                                             num_trials=20,
   >>>                                             reward_attr='accuracy',
   >>>                                             time_attr='epoch',
   >>>                                             grace_period=1)
   >>> scheduler.run()
   >>> scheduler.join_jobs()

   Visiualize the results:

   >>> scheduler.get_training_curves(plot=True)
   >>> ag.done()

.. automodule:: autogluon.core
.. currentmodule:: autogluon.core.space

.. autosummary::
   :nosignatures:
   :toctree: _autogen

   Real
   Int
   Bool
   Categorical
   List
   Dict
   AutoGluonObject

Decorators
~~~~~~~~~~

.. currentmodule:: autogluon.core.decorator

Decorators for customizing AutoGluon. Allows hyperparameter search on user-defined objects and functions.

.. autosummary::
   :nosignatures:
   :toctree: _autogen

   args
   obj
   func

