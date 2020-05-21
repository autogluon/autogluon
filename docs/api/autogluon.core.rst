.. role:: hidden
    :class: hidden-section

autogluon.core
==============

Decorators for customizing AutoGluon to apply hyperparameter-tuning on arbitrary user-defined objects and functions.

.. admonition:: Toy Example

   Create class and function with searchable spaces for hyperparameters `name` and `idx`:

   >>> import autogluon as ag
   >>> @ag.obj(
   ...     name=ag.space.Categorical('auto', 'gluon'),
   ...     idx = ag.space.Int(0, 100),
   ... )
   >>> class myobj:
   ...     def __init__(self, name, idx):
   ...         self.name = name
   ...         self.idx = idx
   ...
   >>> @ag.func(framework=ag.space.Categorical('mxnet', 'pytorch'))
   >>> def myfunc(framework):
   ...     return framework

   Create the objects using decorated class and function as argument of autogluon.args

   >>> @ag.args(
   ...     h=ag.space.Categorical('test', myobj()),
   ...     i = myfunc(),
   ...     )
   >>> def train_fn(args, reporter):
   ...     h, i = args.h, args.i
   ...     assert hasattr(h, 'name') or h == 'test'
   ...     assert i in ['mxnet', 'pytorch']
   ...     reporter(epoch=1, accuracy=0)

   Create a scheduler and run training trials to search for the best values of the hyperparameters:

   >>> scheduler = ag.scheduler.FIFOScheduler(train_fn,
   ...                                        resource={'num_cpus': 2, 'num_gpus': 0},
   ...                                        num_trials=20,
   ...                                        reward_attr='accuracy',
   ...                                        time_attr='epoch')
   >>> scheduler.run()


Core APIs
---------

.. automodule:: autogluon
.. currentmodule:: autogluon

.. autosummary::
   :nosignatures:

   args
   obj
   func

:hidden:`args`
~~~~~~~~~~~~~~

.. autofunction::
   args

:hidden:`obj`
~~~~~~~~~~~~~

.. autofunction::
   obj

:hidden:`func`
~~~~~~~~~~~~~~

.. autofunction::
   func
