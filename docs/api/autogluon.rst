autogluon
---------

Core API in AutoGluon, including search spaces and decorators for customized search configuration
and user-defined objects.

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
   >>>                                             resource={'num_cpus': 2, 'num_gpus': 0},
   >>>                                             num_trials=10,
   >>>                                             reward_attr='accuracy',
   >>>                                             time_attr='epoch',
   >>>                                             grace_period=1)
   >>> scheduler.run()
   >>> scheduler.join_jobs()

   Visiualize the results and exit:

   >>> scheduler.get_training_curves(plot=True)
   >>> ag.done()

   .. image:: https://raw.githubusercontent.com/zhanghang1989/AutoGluonWebdata/master/doc/api/autogluon.1.png
      :width: 400


Search Space
~~~~~~~~~~~~

.. automodule:: autogluon.space
.. currentmodule:: autogluon.space

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


Decorators for customizing AutoGluon. Allows hyperparameter search on user-defined objects and functions.

.. admonition:: Example

   Create dummy class and function with searchable spaces:

   >>> import autogluon as ag
   >>> @ag.obj(
   >>>     name=ag.space.Categorical('auto', 'gluon'),
   >>>     idx = ag.space.Int(0, 100),
   >>> )
   >>> class myobj:
   ...     def __init__(self, name, idx):
   ...         self.name = name
   ...         self.idx = idx
   >>>
   >>> @ag.func(framework=ag.space.Categorical('mxnet', 'pytorch'))
   >>> def myfunc(framework):
   ...     return framework

   Create the objects using decorated class and function as argument of autogluon.args

   >>> @ag.args(
   >>>     h=ag.space.Categorical('test', myobj()),
   >>>     i = myfunc(),
   >>>     )
   >>> def train_fn(args, reporter):
   >>>     h, i = args.h, args.i
   >>>     assert hasattr(h, 'name') or h == 'test'
   >>>     assert i in ['mxnet', 'pytorch']
   >>>     reporter(epoch=e, accuracy=0)

   Create a scheduler and run the dummy experiment:

   >>> scheduler = ag.scheduler.FIFOScheduler(train_fn,
   >>>                                        resource={'num_cpus': 2, 'num_gpus': 0},
   >>>                                        num_trials=20,
   >>>                                        reward_attr='accuracy',
   >>>                                        time_attr='epoch')
   >>> scheduler.run()

   Exit:

   >>> ag.done()

Decorators
~~~~~~~~~~

.. automodule:: autogluon
.. currentmodule:: autogluon

.. autosummary::
   :nosignatures:
   :toctree: _autogen

   args
   obj
   func

