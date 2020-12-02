import pickle
import logging
import threading
import numpy as np
import multiprocessing as mp
import os
import copy

from .fifo import FIFOScheduler
from .hyperband import HyperbandScheduler, _sample_bracket, _sample_bracket, _get_rung_levels, _ARGUMENT_KEYS, _DEFAULT_OPTIONS, _CONSTRAINTS
from .hyperband_stopping import StoppingRungSystem
from .hyperband_promotion import PromotionRungSystem
from .reporter import MODistStatusReporter
from ..utils import load
from ..utils.default_arguments import check_and_merge_defaults, \
    Integer, Boolean, Categorical, filter_by_key

__all__ = ['MOHyperbandScheduler']

logger = logging.getLogger(__name__)

_ARGUMENT_KEYS = {
    'objectives', 'scalarization_options'
}


class MOHyperbandScheduler(HyperbandScheduler):
    r"""Implements different variants of asynchronous Hyperband

    See 'type' for the different variants. One implementation detail is when
    using multiple brackets, task allocation to bracket is done randomly,
    based on a distribution inspired by the synchronous Hyperband case.

    For definitions of concepts (bracket, rung, milestone), see

        Li, Jamieson, Rostamizadeh, Gonina, Hardt, Recht, Talwalkar (2018)
        A System for Massively Parallel Hyperparameter Tuning
        https://arxiv.org/abs/1810.05934

    or

        Tiao, Klein, Lienart, Archambeau, Seeger (2020)
        Model-based Asynchronous Hyperparameter and Neural Architecture Search
        https://arxiv.org/abs/2003.10865

    Note: This scheduler requires both reward and resource (time) to be
    returned by the reporter. Here, resource (time) values must be positive
    int. If time_attr == 'epoch', this should be the number of epochs done,
    starting from 1 (not the epoch number, starting from 0).

    Rung levels and promotion quantiles:

    Rung levels are values of the resource attribute at which stop/go decisions
    are made for jobs, comparing their reward against others at the same level.
    These rung levels (positive, strictly increasing) can be specified via
    `rung_levels`, the largest must be `<= max_t`.
    If `rung_levels` is not given, rung levels are specified by `grace_period`
    and `reduction_factor`:

        [grace_period * (reduction_factor ** j)], j = 0, 1, ...

    This is the default choice for successive halving (Hyperband).
    Note: If `rung_levels` is given, then `grace_period`, `reduction_factor`
    are ignored. If they are given, a warning is logged.

    The rung levels determine the quantiles to be used in the stop/go
    decisions. If rung levels are r_0, r_1, ..., define

        q_j = r_j / r_{j+1}

    q_j is the promotion quantile at rung level r_j. On average, a fraction
    of q_j jobs can continue, the remaining ones are stopped (or paused).
    In the default successive halving case:

        q_j = 1 / reduction_factor    for all j


    Parameters
    ----------
    train_fn : callable
        A task launch function for training.
    args : object, optional
        Default arguments for launching train_fn.
    resource : dict
        Computation resources.  For example, `{'num_cpus':2, 'num_gpus':1}`
    searcher : str or BaseSearcher
        Searcher (get_config decisions). If str, this is passed to
        searcher_factory along with search_options.
    search_options : dict
        If searcher is str, these arguments are passed to searcher_factory.
    checkpoint : str
        If filename given here, a checkpoint of scheduler (and searcher) state
        is written to file every time a job finishes.
        Note: May not be fully supported by all searchers.
    resume : bool
        If True, scheduler state is loaded from checkpoint, and experiment
        starts from there.
        Note: May not be fully supported by all searchers.
    num_trials : int
        Maximum number of jobs run in experiment. One of `num_trials`,
        `time_out` must be given.
    time_out : float
        If given, jobs are started only until this time_out (wall clock time).
        One of `num_trials`, `time_out` must be given.
    objectives : dict
        Dictionary with the names of objectives of interest. The corresponding
        values are allowed to be either "MAX" or "MIN" and indicate if an 
        objective is to be maximized or minimized. The objective data is 
        obtained from the reporter.
    scalarization_options : dict
        Contains arguments for the underlying scalarization algorithm.
    time_attr : str
        Name of resource (or time) attribute in data obtained from reporter.
        Note: The type of resource must be positive int.
    max_t : int
        Maximum resource (see time_attr) to be used for a job. Together with
        `grace_period` and `reduction_factor`, this is used to determine rung
        levels in Hyperband brackets (if `rung_levels` is not given).
        Note: If this is not given, we try to infer its value from `train_fn.args`,
        checking `train_fn.args.epochs` or `train_fn.args.max_t`. If `max_t` is
        given as argument here, it takes precedence.
    grace_period : int
        Minimum resource (see `time_attr`) to be used for a job.
        Ignored if `rung_levels` is given.
    reduction_factor : int (>= 2)
        Parameter to determine rung levels in successive halving (Hyperband).
        Ignored if `rung_levels` is given.
    rung_levels: list of int
        If given, prescribes the set of rung levels to be used. Must contain
        positive integers, strictly increasing. This information overrides
        `grace_period` and `reduction_factor`, but not `max_t`.
        Note that the stop/promote rule in the successive halving scheduler is
        set based on the ratio of successive rung levels.
    brackets : int
        Number of brackets to be used in Hyperband. Each bracket has a different
        grace period, all share max_t and reduction_factor.
        If brackets == 1, we just run successive halving.
    training_history_callback : callable
        Callback function func called every time a job finishes, if at least
        training_history_callback_delta_secs seconds passed since the last
        recent call. The call has the form:
            func(self.training_history, self._start_time)
        Here, self._start_time is time stamp for when experiment started.
        Use this callback to serialize self.training_history after regular
        intervals.
    training_history_callback_delta_secs : float
        See training_history_callback.
    delay_get_config : bool
        If True, the call to searcher.get_config is delayed until a worker
        resource for evaluation is available. Otherwise, get_config is called
        just after a job has been started.
        For searchers which adapt to past data, True should be preferred.
        Otherwise, it does not matter.
    type : str
        Type of Hyperband scheduler:
            stopping:
                A config eval is executed by a single task. The task is stopped
                at a milestone if its metric is worse than a fraction of those
                who reached the milestone earlier, otherwise it continues.
                As implemented in Ray/Tune:
                https://ray.readthedocs.io/en/latest/tune-schedulers.html#asynchronous-hyperband
                See :class:`StoppingRungSystem`.
            promotion:
                A config eval may be associated with multiple tasks over its
                lifetime. It is never terminated, but may be paused. Whenever a
                task becomes available, it may promote a config to the next
                milestone, if better than a fraction of others who reached the
                milestone. If no config can be promoted, a new one is chosen.
                This variant may benefit from pause&resume, which is not directly
                supported here. As proposed in this paper (termed ASHA):
                https://arxiv.org/abs/1810.05934
                See :class:`PromotionRungSystem`.
    dist_ip_addrs : list of str
        IP addresses of remote machines.
    maxt_pending : bool
        Relevant only if a model-based searcher is used.
        If True, register pending config at level max_t whenever a task is started.
        This has a direct effect on the acquisition function (for model-based
        variant), which operates at level max_t. On the other hand, it decreases
        the variance of the latent process there.
    searcher_data : str
        Relevant only if a model-based searcher is used, and if train_fn is such
        that we receive results (from the reporter) at each successive resource
        level, not just at the rung levels.
        Example: For NN tuning and `time_attr` == 'epoch', we receive a result for
        each epoch, but not all epoch values are also rung levels.
        searcher_data determines which of these results are passed to the
        searcher. As a rule, the more data the searcher receives, the better its
        fit, but also the more expensive get_config may become. Choices:
        - 'rungs' (default): Only results at rung levels. Cheapest
        - 'all': All results. Most expensive
        - 'rungs_and_last': Results at rung levels, plus the most recent result.
            This means that in between rung levels, only the most recent result
            is used by the searcher. This is in between
    rung_system_per_bracket : bool
        This concerns Hyperband with brackets > 1. When starting a job for a
        new config, it is assigned a randomly sampled bracket. The larger the
        bracket, the larger the grace period for the config. If
        `rung_system_per_bracket` is True, we maintain separate rung level
        systems for each bracket, so that configs only compete with others
        started in the same bracket. This is the default behaviour of Hyperband.
        If False, we use a single rung level system, so that all configs
        compete with each other. In this case, the bracket of a config only
        determines the initial grace period, i.e. the first milestone at which
        it starts competing with others.
        The concept of brackets in Hyperband is meant to hedge against overly
        aggressive filtering in successive halving, based on low fidelity
        criteria. In practice, successive halving (i.e., `brackets = 1`) often
        works best in the asynchronous case (as implemented here). If
        `brackets > 1`, the hedging is stronger if `rung_system_per_bracket`
        is True.
    random_seed : int
        Random seed for PRNG for bracket sampling

    See Also
    --------
    HyperbandBracketManager

    TODO: ADAPT THIS ONE
    Examples
    --------
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
    >>> scheduler = ag.scheduler.HyperbandScheduler(
    ...     train_fn,
    ...     resource={'num_cpus': 2, 'num_gpus': 0},
    ...     num_trials=20,
    ...     reward_attr='accuracy',
    ...     time_attr='epoch',
    ...     grace_period=1)
    >>> scheduler.run()
    >>> scheduler.join_jobs()
    >>> scheduler.get_training_curves(plot=True)
    """

    def __init__(self, train_fn, **kwargs):
        self._scalarization_options = kwargs["scalarization_options"]
        self._objectives = kwargs["objectives"]
        self._sign_vector = _prepare_sign_vector(self._objectives)
        kwargs["reward_attr"] = "_SCALARIZATION"
        super().__init__(train_fn, **filter_by_key(kwargs, _ARGUMENT_KEYS))


    def add_job(self, task, **kwargs):
        """Adding a training task to the scheduler.

        Args:
            task (:class:`autogluon.scheduler.Task`): a new training task

        Relevant entries in kwargs:
        - bracket: HB bracket to be used. Has been sampled in _promote_config
        - new_config: If True, task starts new config eval, otherwise it promotes
          a config (only if type == 'promotion')
        - elapsed_time: Time stamp
        Only if new_config == False:
        - config_key: Internal key for config
        - resume_from: config promoted from this milestone
        - milestone: config promoted to this milestone (next from resume_from)
        """
        cls = HyperbandScheduler
        if not self._delay_get_config:
            # Wait for resource to become available here, as this has not happened
            # in schedule_next before
            cls.resource_manager._request(task.resources)
        # reporter and terminator
        weights = [_uniform(len(self._objectives)) for _ in range(self._scalarization_options["num_weights"])]
        weights = [w * self._sign_vector for w in weights]
        reporter = MODistStatusReporter(self._objectives, weights ,remote=task.resources.node)
        task.args['reporter'] = reporter

        # We could factor this part out by modifiying the main class
        # --------------------------------------------------------------------------------

        # Register task
        task_key = str(task.task_id)
        with self._hyperband_lock:
            assert task_key not in self._running_tasks, \
                "Task {} is already registered as running".format(task_key)
            self._running_tasks[task_key] = {
                'config': task.args['config'],
                'time_stamp': kwargs['elapsed_time'],
                'bracket': kwargs['bracket'],
                'reported_result': None,
                'keep_case': False}
            first_milestone = self.terminator.on_task_add(task, **kwargs)[-1]
        # Register pending evaluation(s) with searcher
        debug_log = self.searcher.debug_log
        if kwargs.get('new_config', True):
            # Task starts a new config
            next_milestone = first_milestone
            logger.debug("Adding new task (first milestone = {}):\n{}".format(
                next_milestone, task))
            if debug_log is not None:
                # Debug log output
                config_id = debug_log.config_id(task.args['config'])
                msg = "config_id {} starts (first milestone = {})".format(
                    config_id, next_milestone)
                logger.info(msg)
        else:
            # Promotion of config
            # This is a signal towards train_fn, which can be used for pause
            # & resume (given that train_fn checkpoints model state): Access
            # in train_fn as args.scheduler.resume_from
            if 'scheduler' not in task.args['args']:
                task.args['args']['scheduler'] = dict()
            task.args['args']['scheduler']['resume_from'] = kwargs['resume_from']
            next_milestone = kwargs['milestone']
            logger.debug("Promotion task (next milestone = {}):\n{}".format(
                next_milestone, task))
            if debug_log is not None:
                # Debug log output
                config_id = debug_log.config_id(task.args['config'])
                msg = "config_id {} promoted from {} (next milestone = {})".format(
                    config_id, kwargs['resume_from'], next_milestone)
                logger.info(msg)
        self.searcher.register_pending(
            task.args['config'], milestone=next_milestone)
        if self.maxt_pending and next_milestone != self.max_t:
            # Also introduce pending evaluation for resource max_t
            self.searcher.register_pending(
                task.args['config'], milestone=self.max_t)

        # main process
        job = cls._start_distributed_job(task, cls.resource_manager)
        # reporter thread
        rp = threading.Thread(
            target=self._run_reporter,
            args=(task, job, reporter),
            daemon=False)
        rp.start()
        task_dict = self._dict_from_task(task)
        task_dict.update({'Task': task, 'Job': job, 'ReporterThread': rp})
        # Checkpoint thread. This is also used for training_history
        # callback
        if self._checkpoint is not None or \
                self.training_history_callback is not None:
            self._add_checkpointing_to_job(job)
        with self.LOCK:
            self.scheduled_tasks.append(task_dict)


def _prepare_sign_vector(objectives):
    """Generates a numpy vector which can be used to flip the signs of the objectives values
    which are intended to be minimized.

    Parameters
    ----------
    objectives: dict
        The dictionary keys name the objectives of interest. The associated values can be either
        "MIN" or "MAX" and indicate if an objective is to be minimized or maximized.

    Returns
    ----------
    sign_vector: np.array
        A numpy array containing 1 for objectives to be maximized and -1 for objectives to be
        minimized.
    """
    converter = {
        "MIN": -1.0,
        "MAX": 1.0
    }
    try:
        sign_vector = np.array([converter[objectives[k]] for k in objectives])
    except KeyError:
        raise ValueError("Error, in conversion of objective dict. Allowed values are 'MIN' and 'MAX'")
    return sign_vector


def _uniform(dim):
    """Samples a point uniformly at random from the unit simplex using the Kraemer Algorithm
    The algorithm is described here: https://www.cs.cmu.edu/~nasmith/papers/smith+tromble.tr04.pdf
    
    Parameters
    ----------
    dim: int
        Dimension of the unit simplex to sample from.
    
    Returns:
    sample: np.array
         A point sampled uniformly from the unit simplex.
    """
    uni = np.random.uniform(size=(dim))
    uni = np.sort(uni)
    sample =  np.diff(uni, prepend=0) / uni[-1]
    assert sum(sample) - 1 < 1e-6, "Error in weight sampling."
    return np.array(sample)
