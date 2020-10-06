.. role:: hidden
    :class: hidden-section

autogluon.core.space
====================

Search space of possible hyperparameter values to consider.

.. admonition:: Example

   Define a dummy training function with searchable spaces for hyperparameters `lr` and `wd`:

   >>> import numpy as np
   >>> import autogluon.core as ag
   >>> @ag.args(
   >>>     lr=ag.space.Real(1e-3, 1e-2, log=True),
   ...     wd=ag.space.Real(1e-3, 1e-2),
   ...     epochs=10)
   >>> def train_fn(args, reporter):
   ...     print('lr: {}, wd: {}'.format(args.lr, args.wd))
   ...     for e in range(args.epochs):
   ...         dummy_accuracy = 1 - np.power(1.8, -np.random.uniform(e, 2*e))
   ...         reporter(epoch=e+1, accuracy=dummy_accuracy, lr=args.lr, wd=args.wd)

   Create a scheduler to manage training jobs and begin hyperparameter tuning with the provided search space:

   >>> scheduler = ag.scheduler.HyperbandScheduler(train_fn,
   >>>                                             resource={'num_cpus': 2, 'num_gpus': 0},
   >>>                                             num_trials=10,
   >>>                                             reward_attr='accuracy',
   >>>                                             time_attr='epoch',
   >>>                                             grace_period=1)
   >>> scheduler.run()
   >>> scheduler.join_jobs()

   Visualize the results:

   >>> scheduler.get_training_curves(plot=True)

   .. image:: https://raw.githubusercontent.com/zhanghang1989/AutoGluonWebdata/master/doc/api/autogluon.1.png
      :width: 400


Search Space
------------

.. automodule:: autogluon.core.space
.. currentmodule:: autogluon.core.space

.. autosummary::
   :nosignatures:

   Real
   Int
   Bool
   Categorical
   List
   Dict
   AutoGluonObject

:hidden:`Real`
~~~~~~~~~~~~~~

.. autoclass:: Real
   :members:

    .. rubric:: Methods

    .. autoautosummary:: Real
        :methods:

:hidden:`Int`
~~~~~~~~~~~~~

.. autoclass:: Int
   :members:

    .. rubric:: Methods

    .. autoautosummary:: Int
        :methods:

:hidden:`Bool`
~~~~~~~~~~~~~~

.. autoclass:: Bool
   :members:

    .. rubric:: Methods

    .. autoautosummary:: Bool
        :methods:

:hidden:`Categorical`
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Categorical
   :members:

    .. rubric:: Methods

    .. autoautosummary:: Categorical
        :methods:

:hidden:`List`
~~~~~~~~~~~~~~

.. autoclass:: List
   :members:

    .. rubric:: Methods

    .. autoautosummary:: List
        :methods:

:hidden:`Dict`
~~~~~~~~~~~~~~

.. autoclass:: Dict
   :members:

    .. rubric:: Methods

    .. autoautosummary:: Dict
        :methods:

:hidden:`AutoGluonObject`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AutoGluonObject
   :members:

    .. rubric:: Methods

    .. autoautosummary:: AutoGluonObject
        :methods:
