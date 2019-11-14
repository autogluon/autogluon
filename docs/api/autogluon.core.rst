.. role:: hidden
    :class: hidden-section

autogluon.core
==============

Decorators for customizing AutoGluon. Allows hyperparameter search on user-defined objects and functions.

.. admonition:: Example

   Create dummy class and function with searchable spaces:

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
   ...     reporter(epoch=e, accuracy=0)

   Create a scheduler and run the dummy experiment:

   >>> scheduler = ag.scheduler.FIFOScheduler(train_fn,
   ...                                        resource={'num_cpus': 2, 'num_gpus': 0},
   ...                                        num_trials=20,
   ...                                        reward_attr='accuracy',
   ...                                        time_attr='epoch')
   >>> scheduler.run()

   Exit:

   >>> ag.done()

Core APIs
---------

.. automodule:: autogluon
.. currentmodule:: autogluon

.. autosummary::
   :nosignatures:

   args
   obj
   func
   done

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

:hidden:`done`
~~~~~~~~~~~~~~

.. autofunction::
   done

