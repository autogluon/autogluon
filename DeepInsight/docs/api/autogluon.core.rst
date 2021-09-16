.. role:: hidden
    :class: hidden-section

autogluon.core
==============

Decorators are designed to apply hyperparameter-tuning on arbitrary user-defined search space.

.. admonition:: Toy Example

   Create a training function decorated with searchable hyperparameters space:

   >>> @ag.args(
   ...     nested=ag.space.Categorical(
   ...         'test',
   ...         ag.space.Dict(
   ...             name=ag.space.Categorical('auto', 'gluon'),
   ...             idx = ag.space.Int(0, 100),
   ...         )
   ...     ),
   ...     obj_args = ag.space.Dict(
   ...         name=ag.space.Categorical('auto', 'gluon'),
   ...         idx = ag.space.Int(0, 100),
   ...     ),
   ...     fn_args=ag.space.Categorical('mxnet', 'pytorch'))
   >>> def train_fn(args, reporter):
   ...         # Wrap parameterizable classes and functions inside train_fn
   ...         # to ensure they are portable for distributed training
   ...         class MyObj:
   ...             def __init__(self, name, idx):
   ...                 self.name = name
   ...                 self.idx = idx
   ...
   ...         def my_func(framework):
   ...             return framework
   ...
   ...         obj = MyObj(**args.obj_args)
   ...         func_result = my_func(args.fn_args)
   ...         nested = args.nested
   ...
   ...         assert hasattr(nested, 'name') or nested == 'test'
   ...         assert func_result in ['mxnet', 'pytorch']
   ...         assert obj.name in ['auto', 'gluon']
   ...         assert obj.idx >=0 and obj.idx <=100
   ...
   ...         reporter(epoch=1, accuracy=0)

   Create a scheduler and run training trials to search for the best values of the hyperparameters:

   >>> scheduler = ag.scheduler.FIFOScheduler(train_fn,
   ...                                        resource={'num_cpus': 2, 'num_gpus': 0},
   ...                                        num_trials=20,
   ...                                        reward_attr='accuracy',
   ...                                        time_attr='epoch')
   >>> scheduler.run()


Core APIs
---------

.. automodule:: autogluon.core
.. currentmodule:: autogluon.core

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
