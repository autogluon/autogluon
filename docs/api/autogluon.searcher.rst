.. role:: hidden
    :class: hidden-section

autogluon.searcher
------------------

.. admonition:: Example
   Define a dummy training function with searchable spaces:

   >>> import numpy as np
   >>> import autogluon as ag
   >>> @ag.args(
   ...     lr=ag.space.Real(1e-3, 1e-2, log=True),
   ...     wd=ag.space.Real(1e-3, 1e-2))
   >>> def train_fn(args, reporter):
   ...     print('lr: {}, wd: {}'.format(args.lr, args.wd))
   ...     for e in range(10):
   ...         dummy_accuracy = 1 - np.power(1.8, -np.random.uniform(e, 2*e))
   ...         reporter(epoch=e, accuracy=dummy_accuracy, lr=args.lr, wd=args.wd)

   Create a searcher and sample one configuration

   >>> searcher = ag.searcher.SKoptSearcher(train_fn.cs)
   >>> searcher.get_config()
   {'lr': 0.0031622777, 'wd': 0.0055}

   Create a scheduler and run the dummy experiment:

   >>> scheduler = ag.scheduler.FIFOScheduler(train_fn,
   ...                                        searcher = searcher,
   ...                                        resource={'num_cpus': 2, 'num_gpus': 0},
   ...                                        num_trials=10,
   ...                                        reward_attr='accuracy',
   ...                                        time_attr='epoch')
   >>> scheduler.run()

   Visiualize the results and exit:

   >>> scheduler.get_training_curves(plot=True)

   .. image:: https://raw.githubusercontent.com/zhanghang1989/AutoGluonWebdata/master/doc/api/autogluon.searcher.1.png
      :width: 400

Searchers
~~~~~~~~~

.. currentmodule:: autogluon.searcher

.. autosummary::
   :nosignatures:

   GridSearcher
   RandomSearcher
   SKoptSearcher
   RLSearcher

:hidden:`GridSearcher`
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GridSearcher
   :members:
   :inherited-members:

    .. rubric:: Methods

    .. autoautosummary:: GridSearcher
        :methods:

:hidden:`RandomSearcher`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RandomSearcher
   :members:
   :inherited-members:

    .. rubric:: Methods

    .. autoautosummary:: RandomSearcher
        :methods:


:hidden:`SKoptSearcher`
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SKoptSearcher
   :members:
   :inherited-members:

    .. rubric:: Methods

    .. autoautosummary:: SKoptSearcher
        :methods:

:hidden:`RLSearcher`
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RLSearcher
   :members:
   :inherited-members:

    .. rubric:: Methods

    .. autoautosummary:: RLSearcher
        :methods:
