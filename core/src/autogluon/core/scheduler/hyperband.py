import pickle
import logging
import numpy as np
import multiprocessing as mp
import os
import copy

from .fifo import FIFOScheduler
from .hyperband_stopping import StoppingRungSystem
from .hyperband_promotion import PromotionRungSystem
from ..utils import load
from ..utils.default_arguments import check_and_merge_defaults, \
    Integer, Boolean, Categorical, filter_by_key

__all__ = ['HyperbandScheduler',
           'HyperbandBracketManager']

logger = logging.getLogger(__name__)


_ARGUMENT_KEYS = {
    'max_t', 'grace_period', 'reduction_factor', 'brackets', 'type',
    'searcher_data', 'do_snapshots', 'rung_system_per_bracket',
    'keep_size_ratios', 'random_seed', 'rung_levels'}

_DEFAULT_OPTIONS = {
    'resume': False,
    'grace_period': 1,
    'reduction_factor': 3,
    'brackets': 1,
    'type': 'stopping',
    'searcher_data': 'rungs',
    'do_snapshots': False,
    'rung_system_per_bracket': True,
    'random_seed': 31415927,
    'rung_levels': None}

_CONSTRAINTS = {
    'resume': Boolean(),
    'max_t': Integer(1, None),
    'grace_period': Integer(1, None),
    'reduction_factor': Integer(2, None),
    'brackets': Integer(1, None),
    'type': Categorical(('stopping', 'promotion')),
    'searcher_data': Categorical(
        ('rungs', 'all', 'rungs_and_last')),
    'do_snapshots': Boolean(),
    'rung_system_per_bracket': Boolean(),
    'random_seed': Integer(0, None)}


class HyperbandScheduler(FIFOScheduler):
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
        Moreover, we also stop jobs after time_out has passed, when they
        report a result.
        One of `num_trials`, `time_out` must be given.
    reward_attr : str
        Name of reward (i.e., metric to maximize) attribute in data obtained
        from reporter
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
        Callback function func called every time a result is added to
        training_history, if at least training_history_callback_delta_secs
        seconds passed since the last recent call.
        See _add_training_result for the signature of this callback function.
        Use this callback to serialize self.training_history after regular
        intervals.
    training_history_callback_delta_secs : float
        See training_history_callback.
    training_history_searcher_info : bool
        If True, information about the current state of the searcher is added
        to every reported_result before added to training_history. This info
        includes in particular the current hyperparameters of the surrogate
        model of the searcher, as well as the dataset size.
    delay_get_config : bool
        If True, the call to searcher.get_config is delayed until a worker
        resource for evaluation is available. Otherwise, get_config is called
        just after a job has been started.
        For searchers which adapt to past data, True should be preferred.
        Otherwise, it does not matter.
    stop_jobs_after_time_out : bool
        Relevant only if `time_out` is used. If True, jobs which report a
        metric are stopped once `time_out` has passed. Otherwise, such jobs
        are allowed to continue until the end, or until stopped for other
        reasons. The latter can mean an experiment runs far longer than
        `time_out`.
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
    do_snapshots : bool
        Support snapshots? If True, a snapshot of all running tasks and rung
        levels is returned by _promote_config. This snapshot is passed to the
        searcher in get_config.
        Note: Currently, only the stopping variant supports snapshots.

    See Also
    --------
    HyperbandBracketManager


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
        # Setting max_t:
        # A well-written train_fn reveals its max_t value. We check fields in
        # train_fn.args: epochs, max_t.
        # In any case, the max_t argument takes precedence. If it is None, we use
        # the one inferred from train_fn.args. If neither is given, we raise an
        # exception
        inferred_max_t = self._infer_max_t(train_fn.args)
        max_t = kwargs.get('max_t')
        if max_t is not None:
            if inferred_max_t is not None and max_t != inferred_max_t:
                logger.warning(
                    "max_t = {} is different from the value {} inferred from train_fn.args (train_fn.args.epochs, train_fn.args.max_t)".format(max_t, inferred_max_t))
        else:
            assert inferred_max_t is not None, \
                "Either max_t must be specified, or it has to be specified via train_fn (as train_fn.args.epochs or train_fn.args.max_t)"
            logger.info("max_t = {}, as inferred from train_fn.args".format(
                inferred_max_t))
            max_t = inferred_max_t
        # Deprecated kwargs
        deprecated_keys = ('keep_size_ratios', 'maxt_pending')
        for k in deprecated_keys:
            if k in kwargs:
                logger.warning("'{}' is deprecated, will be ignored".format(k))
                del kwargs[k]
        # If rung_levels is given, grace_period and reduction_factor are ignored
        rung_levels = kwargs.get('rung_levels')
        if rung_levels is not None:
            assert isinstance(rung_levels, list)
            if ('grace_period' in kwargs) or ('reduction_factor' in kwargs):
                logger.warning(
                    "Since rung_levels is given, the values grace_period = "
                    "{} and reduction_factor = {} are ignored!".format(
                        kwargs['grace_period'], kwargs['reduction_factor']))
        # Check values and impute default values (only for arguments new to
        # this class)
        kwargs = check_and_merge_defaults(
            kwargs, set(), _DEFAULT_OPTIONS, _CONSTRAINTS,
            dict_name='scheduler_options')
        resume = kwargs['resume']
        scheduler_type = kwargs['type']
        supported_types = {'stopping', 'promotion'}
        assert scheduler_type in supported_types, \
            "type = '{}' not supported, must be in {}".format(
                scheduler_type, supported_types)
        rung_levels = _get_rung_levels(
            rung_levels, grace_period=kwargs['grace_period'],
            reduction_factor=kwargs['reduction_factor'], max_t=max_t)
        brackets = kwargs['brackets']
        do_snapshots = kwargs['do_snapshots']
        assert (not do_snapshots) or (scheduler_type == 'stopping'), \
            "Snapshots are supported only for type = 'stopping'"
        rung_system_per_bracket = kwargs['rung_system_per_bracket']

        # Adjoin information about scheduler to search_options
        search_options = kwargs.get('search_options')
        if search_options is None:
            _search_options = dict()
        else:
            _search_options = search_options.copy()
        _search_options['scheduler'] = 'hyperband_{}'.format(scheduler_type)
        _search_options['min_epochs'] = rung_levels[0]
        _search_options['max_epochs'] = max_t
        kwargs['search_options'] = _search_options
        # Pass resume=False here. Resume needs members of this object to be
        # created
        kwargs['resume'] = False
        super().__init__(
            train_fn=train_fn, **filter_by_key(kwargs, _ARGUMENT_KEYS))

        self.max_t = max_t
        self.scheduler_type = scheduler_type
        self.terminator = HyperbandBracketManager(
            scheduler_type, self._time_attr, self._reward_attr, max_t,
            rung_levels, brackets, rung_system_per_bracket,
            kwargs['random_seed'])
        self.do_snapshots = do_snapshots
        self.searcher_data = kwargs['searcher_data']
        # Maintains a snapshot of currently running tasks, needed by several
        # features (for example, searcher_data == 'rungs_and_last', or for
        # providing a snapshot to the searcher).
        # Maps str(task_id) to dict, with fields:
        # - config
        # - time_stamp: Time when task was started, or when last recent
        #       result was reported
        # - reported_result: Last recent reported result, or None (task was
        #       started, but did not report anything yet.
        #       Note: Only contains attributes self._reward_attr and
        #       self._time_attr).
        # - bracket: Bracket number
        # - keep_case: Boolean flag. Relevant only if searcher_data ==
        #   'rungs_and_last'. See _run_reporter
        self._running_tasks = dict()
        # This lock protects both _running_tasks and terminator, the latter
        # does not define its own lock
        self._hyperband_lock = mp.Lock()
        if resume:
            checkpoint = kwargs.get('checkpoint')
            assert checkpoint is not None, \
                "Need checkpoint to be set if resume = True"
            if os.path.isfile(checkpoint):
                self.load_state_dict(load(checkpoint))
            else:
                msg = f'checkpoint path {checkpoint} is not available for resume.'
                logger.exception(msg)
                raise FileExistsError(msg)

    @staticmethod
    def _infer_max_t(args):
        if hasattr(args, 'epochs'):
            return args.epochs
        elif hasattr(args, 'max_t'):
            return args.max_t
        else:
            return None

    def on_task_add(self, task, **kwargs):
        """
        Called when new task is added. Register new task, inform searcher
        (pending evaluation) and train_fn (resume_from, checkpointing).

        Relevant entries in kwargs:
        - bracket: HB bracket to be used. Has been sampled in _promote_config
        - new_config: If True, task starts new config eval, otherwise it promotes
          a config (only if type == 'promotion')
        - elapsed_time: Time stamp
        Only if new_config == False:
        - config_key: Internal key for config
        - resume_from: config promoted from this milestone
        - milestone: config promoted to this milestone (next from resume_from)

        :param task:
        :param kwargs:

        """
        # Register task
        task_key = str(task.task_id)
        config = task.args['config']
        debug_log = self.searcher.debug_log
        config_id = debug_log.config_id(config) if debug_log else None
        with self._hyperband_lock:
            assert task_key not in self._running_tasks, \
                "Task {} is already registered as running".format(task_key)
            self._running_tasks[task_key] = {
                'config': config,
                'time_stamp': kwargs['elapsed_time'],
                'bracket': kwargs['bracket'],
                'reported_result': None,
                'keep_case': False}
            first_milestone = self.terminator.on_task_add(task, **kwargs)[-1]
        if kwargs.get('new_config', True):
            # Task starts a new config
            next_milestone = first_milestone
            if debug_log is not None:
                # Debug log output
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
            resume_from = kwargs['resume_from']
            task.args['args']['scheduler']['resume_from'] = resume_from
            next_milestone = kwargs['milestone']
            if debug_log is not None:
                # Debug log output
                msg = "config_id {} promoted from {} (next milestone = {})".format(
                    config_id, resume_from, next_milestone)
                logger.info(msg)
        # Register pending evaluation(s) with searcher
        self.searcher.register_pending(config, milestone=next_milestone)
        # Relay config_id to train_fn (for debug logging)
        if debug_log is not None:
            task.args['args']['config_id'] = config_id

    def _class_for_add_job(self):
        return HyperbandScheduler

    def _update_searcher_internal(self, task, result):
        config = task.args['config']
        if self.searcher_data == 'rungs_and_last':
            # Remove last recently added result for this task. This is not
            # done if it fell on a rung level (i.e., `keep_case` is True)
            with self._hyperband_lock:
                task_record = self._running_tasks[str(task.task_id)]
                if (task_record['reported_result'] is not None) and \
                        (not task_record['keep_case']):
                    rem_result = task_record['reported_result']
                    self.searcher.remove_case(config, **rem_result)
        self.searcher.update(config, **result)

    def _update_searcher(self, task, result, task_info):
        """
        Updates searcher with `result` (depending on `searcher_data`), and
        registers pending configs with searcher.

        :param task:
        :param result: Record obtained from reporter
        :param task_info: Info from self.terminator.on_task_report
        :return: Has searcher been updated?

        """
        task_continues = task_info['task_continues']
        milestone_reached = task_info['milestone_reached']
        did_update = False
        if self.searcher_data == 'rungs':
            if milestone_reached:
                # Update searcher with intermediate result
                self._update_searcher_internal(task, result)
                did_update = True
                if task_continues:
                    next_milestone = task_info.get('next_milestone')
                    if next_milestone is not None:
                        self.searcher.register_pending(
                            task.args['config'], milestone=next_milestone)
        elif not task_info.get('ignore_data', False):
            # All results are reported to the searcher, except if
            # task_info['ignore_data'] is True. The latter happens only for
            # tasks running promoted configs. In this case, we may receive
            # reports before the first milestone is reached, which should not
            # be passed to the searcher (they'd duplicate earlier
            # datapoints).
            # See also header comment of PromotionRungSystem.
            self._update_searcher_internal(task, result)
            did_update = True
            # Since all results are reported, the next report for this task
            # will be for resource + 1.
            # NOTE: This assumes that results are reported for all successive
            # resource levels (int). If any resource level is skipped,
            # there may be left-over pending candidates, which will be
            # removed once the task finishes.
            if task_continues:
                self.searcher.register_pending(
                    task.args['config'],
                    milestone=int(result[self._time_attr]) + 1)
        return did_update

    def on_task_report(self, task, result):
        """
        Called by reporter thread once a new result is reported.

        :param task:
        :param result:
        :return: Should reporter move on? Otherwise, it terminates

        """
        task_key = str(task.task_id)
        debug_log = self.searcher.debug_log
        config = task.args['config']
        config_id = debug_log.config_id(config) if debug_log else None
        task_continues = None
        if 'traceback' in result:
            # Evaluation has failed
            logger.critical(result['traceback'])
            self.searcher.evaluation_failed(config, **result)
            if debug_log is not None:
                msg = "config_id {}: Evaluation failed:\n{}".format(
                    config_id, result['traceback'])
                logger.info(msg)
            task_continues = False
        elif result.get('done', False):
            task_continues = False
        # Time since start of experiment
        elapsed_time = self._elapsed_time()
        # If we are past self.time_out, we want to stop the job
        if self._stop_jobs_after_time_out and elapsed_time > self.time_out:
            if debug_log is not None:
                msg = "config_id {}: Terminating because elapsed_time = {} > {} = self.time_out".format(
                    config_id, elapsed_time, self.time_out)
                logger.info(msg)
            task_continues = False
        elif len(result) == 0:
            # An empty dict should just be skipped
            if debug_log is not None:
                msg = "config_id {}: Skipping empty dict received from reporter".format(
                    config_id)
                logger.info(msg)
            task_continues = True

        if task_continues is None:
            # Call before _add_training_results, since we may be able to report
            # extra information from the bracket:
            with self._hyperband_lock:
                task_info = self.terminator.on_task_report(task, result)
            task_continues = task_info['task_continues']
            milestone_reached = task_info['milestone_reached']
            # Append extra information to result
            result['bracket'] = task_info['bracket_id']
            if not task_continues:
                result['terminated'] = True
            self._append_extra_searcher_info(result)
            # Pass result to training_history
            self._add_training_result(task.task_id, result, config=config)
            # Update searcher and register pending
            self._update_searcher(task, result, task_info)
            # Change snapshot entry for task
            # Note: This must not be done above, because what _update_searcher
            # is doing, depends on the entry *before* its update here.
            with self._hyperband_lock:
                # Note: result may contain all sorts of extra info.
                # All we need to maintain in the snapshot are reward and
                # resource level.
                # 'keep_case' entry (only used if searcher_data ==
                # 'rungs_and_last'): The result is kept in the dataset iff
                # milestone_reached == True (i.e., we are at a rung level).
                # Otherwise, it is removed once _update_searcher is called for
                # the next recent result.
                self._running_tasks[task_key].update({
                    'time_stamp': elapsed_time,
                    'reported_result': {
                        self._reward_attr: result[self._reward_attr],
                        self._time_attr: result[self._time_attr]},
                    'keep_case': milestone_reached})

            if debug_log is not None:
                resource = int(result[self._time_attr])
                if not task_continues:
                    if self.scheduler_type == 'stopping' or resource >= self.max_t:
                        act_str = 'Terminating'
                    else:
                        act_str = 'Pausing'
                    msg = "config_id {}: {} evaluation at {}".format(
                        config_id, act_str, resource)
                    logger.info(msg)
                elif milestone_reached:
                    msg = "config_id {}: Reaches {}, continues".format(
                        config_id, resource)
                    next_milestone = task_info.get('next_milestone')
                    if next_milestone is not None:
                        msg += " to {}".format(next_milestone)
                    logger.info(msg)

        if not task_continues:
            with self._hyperband_lock:
                self.terminator.on_task_remove(task)
                # Cleanup
                if task_key in self._running_tasks:
                    del self._running_tasks[task_key]
        return task_continues

    def _run_reporter(self, task, task_job, reporter):
        while not task_job.done():
            reported_result = reporter.fetch()
            if self.on_task_report(task, reported_result):
                reporter.move_on()
            else:
                reporter.terminate()
                break

    def _snapshot_tasks(self, bracket_id):
        # If all brackets share a single rung level system, then all
        # running jobs have to be taken into account, otherwise only
        # those jobs running in the same bracket
        all_running = not self.terminator._rung_system_per_bracket
        tasks = dict()
        for k, v in self._running_tasks.items():
            if all_running or (v['bracket'] == bracket_id):
                reported_result = v['reported_result']
                level = 0 if reported_result is None \
                    else reported_result[self._time_attr]
                # It is possible to have tasks in _running_tasks which have
                # reached self.max_t. These must not end up in the snapshot
                if level < self.max_t:
                    tasks[k] = {
                        'config': v['config'],
                        'time': v['time_stamp'],
                        'level': level}
        return tasks

    # Snapshot (in extra_kwargs['snapshot']):
    # - max_resource
    # - reduction_factor
    # - tasks: Info about running tasks in bracket bracket_id (or, if
    #   brackets share the same rung level system, all running tasks):
    #   dict(task_id) -> dict:
    #   - config: config as dict
    #   - time: Time when task was started, or when last recent result was
    #     reported
    #   - level: Level of last recent result report, or 0 if no reports yet
    # - rungs: Metric values at rung levels in bracket bracket_id:
    #   List of (rung_level, metric_dict), where metric_dict has entries
    #   task_id: metric_value. Note that entries are sorted in decreasing order
    #   w.r.t. rung_level.
    def _promote_config(self):
        with self._hyperband_lock:
            config, extra_kwargs = self.terminator.on_task_schedule()
            if self.do_snapshots:
                # Append snapshot
                bracket_id = extra_kwargs['bracket']
                extra_kwargs['snapshot'] = {
                    'tasks': self._snapshot_tasks(bracket_id),
                    'rungs': self.terminator.snapshot_rungs(bracket_id),
                    'max_resource': self.max_t}
            debug_log = self.searcher.debug_log
            if (debug_log is not None) and (config is not None):
                # Debug log output
                config_id = debug_log.config_id(config)
                msg = "config_id {}: Promotion from {} to {}".format(
                    config_id, extra_kwargs['resume_from'],
                    extra_kwargs['milestone'])
                logger.info(msg)
            return config, extra_kwargs

    def state_dict(self, destination=None, no_fifo_lock=False):
        """Returns a dictionary containing a whole state of the scheduler

        Examples
        --------
        >>> import autogluon.core as ag
        >>> ag.save(scheduler.state_dict(), 'checkpoint.ag')
        """
        destination = super().state_dict(
            destination, no_fifo_lock=no_fifo_lock)
        # Note: _running_tasks is not part of the state to be checkpointed.
        # The assumption is that if an experiment is resumed from a
        # checkpoint, tasks which did not finish at the checkpoint, are not
        # restarted
        with self._hyperband_lock:
            destination['terminator'] = pickle.dumps(self.terminator)
        return destination

    def load_state_dict(self, state_dict):
        """Load from the saved state dict.

        Examples
        --------
        >>> import autogluon.core as ag
        >>> scheduler.load_state_dict(ag.load('checkpoint.ag'))
        """
        with self._hyperband_lock:
            assert len(self._running_tasks) == 0, \
                "load_state_dict must only be called as part of scheduler construction"
            super().load_state_dict(state_dict)
            # Note: _running_tasks is empty from __init__, it is not recreated,
            # since running tasks are not part of the checkpoint
            self.terminator = pickle.loads(state_dict['terminator'])
            logger.info('Loading Terminator State {}'.format(self.terminator))

    def __repr__(self):
        reprstr = self.__class__.__name__ + '(' +  \
            'terminator: ' + str(self.terminator)
        return reprstr


def _sample_bracket(num_brackets, rung_levels, random_state=None):
    # Brackets are sampled in proportion to the number of configs started
    # in synchronous Hyperband in each bracket
    if num_brackets > 1:
        smax_plus1 = len(rung_levels)
        assert num_brackets <= smax_plus1
        probs = np.array([
            smax_plus1 / ((smax_plus1 - s) * rung_levels[s])
            for s in range(num_brackets)])
        normalized = probs / probs.sum()
        if random_state is None:
            random_state = np.random
        return random_state.choice(num_brackets, p=normalized)
    else:
        return 0


def _is_positive_int(x):
    return int(x) == x and x >= 1


def _get_rung_levels(rung_levels, grace_period, reduction_factor, max_t):
    if rung_levels is not None:
        assert isinstance(rung_levels, list) and len(rung_levels) > 1, \
            "rung_levels must be list of size >= 2"
        assert all(_is_positive_int(x) for x in rung_levels), \
            "rung_levels must be list of positive integers"
        rung_levels = [int(x) for x in rung_levels]
        assert all(x < y for x, y in zip(rung_levels, rung_levels[1:])), \
            "rung_levels must be strictly increasing sequence"
        assert rung_levels[-1] <= max_t, \
            "Last entry of rung_levels ({}) must be <= max_t ({})".format(
                rung_levels[-1], max_t)
    else:
        # Rung levels given by grace_period, reduction_factor, max_t
        assert _is_positive_int(grace_period)
        assert _is_positive_int(reduction_factor)
        assert _is_positive_int(max_t)
        assert max_t > grace_period, \
            "max_t ({}) must be greater than grace_period ({})".format(
                max_t, grace_period)
        rf = reduction_factor
        min_t = grace_period
        max_rungs = int(np.log(max_t / min_t) / np.log(rf) + 1)
        rung_levels = [min_t * rf ** k for k in range(max_rungs)]
        assert rung_levels[-1] <= max_t  # Sanity check
        assert len(rung_levels) >= 2, \
            "grace_period = {}, reduction_factor = {}, max_t = {} leads to single rung level only".format(
                grace_period, reduction_factor, max_t)

    return rung_levels


class HyperbandBracketManager(object):
    """Hyperband Manager

    Maintains rung level systems for range of brackets. Differences depending
    on `scheduler_type` ('stopping', 'promotion') manifest themselves mostly
    at the level of the rung level system itself.

    For `scheduler_type` == 'stopping', see :class:`StoppingRungSystem`.
    For `scheduler_type` == 'promotion', see :class:`PromotionRungSystem`.

    Args:
        scheduler_type : str
            See HyperbandScheduler.
        time_attr : str
            See HyperbandScheduler.
        reward_attr : str
            See HyperbandScheduler.
        max_t : int
            See HyperbandScheduler.
        rung_levels : list[int]
            See HyperbandScheduler. If `rung_levels` is not given there, the
            default rung levels based on `grace_period` and `reduction_factor`
            are used.
        brackets : int
            See HyperbandScheduler.
        rung_system_per_bracket : bool
            See HyperbandScheduler.
        random_seed : int
            Random seed for bracket sampling

    """
    def __init__(
            self, scheduler_type, time_attr, reward_attr, max_t, rung_levels,
            brackets, rung_system_per_bracket, random_seed):
        self._scheduler_type = scheduler_type
        self._reward_attr = reward_attr
        self._time_attr = time_attr
        self._max_t = max_t
        self.rung_levels = copy.copy(rung_levels)
        self._rung_system_per_bracket = rung_system_per_bracket
        # Maps str(task_id) -> bracket_id
        self._task_info = dict()
        max_num_brackets = len(rung_levels)
        self.num_brackets = min(brackets, max_num_brackets)
        num_systems = self.num_brackets if rung_system_per_bracket else 1
        if scheduler_type == 'stopping':
            rs_type = StoppingRungSystem
        else:
            rs_type = PromotionRungSystem
        rung_levels_plus_maxt = rung_levels[1:] + [max_t]
        # Promotion quantiles: q_j = r_j / r_{j+1}
        promote_quantiles = [
            x / y for x, y in zip(rung_levels, rung_levels_plus_maxt)]
        self._rung_systems = [
            rs_type(rung_levels[s:], promote_quantiles[s:], max_t)
            for s in range(num_systems)]
        self.random_state = np.random.RandomState(random_seed)

    def _get_rung_system_for_bracket_id(self, bracket_id):
        if self._rung_system_per_bracket:
            sys_id = bracket_id
            skip_rungs = 0
        else:
            sys_id = 0
            skip_rungs = bracket_id
        return self._rung_systems[sys_id], skip_rungs

    def _get_rung_system(self, task_id):
        bracket_id = self._task_info[str(task_id)]
        rung_sys, skip_rungs = self._get_rung_system_for_bracket_id(bracket_id)
        return rung_sys, bracket_id, skip_rungs

    def on_task_add(self, task, **kwargs):
        """
        Called when new task is started.

        Since the bracket has already been sampled in on_task_schedule,
        not much is done here.
        We return the list of milestones for this bracket in reverse
        (decreasing) order. The first entry is max_t, even if it is
        not a milestone in the bracket. This list contains the resource
        levels the task would reach if it ran to max_t without being stopped.

        :param task: Only task.task_id is used
        :return: List of milestones in decreasing order, where max_t is first
        """
        assert 'bracket' in kwargs
        bracket_id = kwargs['bracket']
        self._task_info[str(task.task_id)] = bracket_id
        rung_sys, skip_rungs = self._get_rung_system_for_bracket_id(bracket_id)
        rung_sys.on_task_add(task, skip_rungs=skip_rungs, **kwargs)
        milestones = rung_sys.get_milestones(skip_rungs)
        if milestones[0] < self._max_t:
            milestones.insert(0, self._max_t)
        return milestones

    def on_task_report(self, task, result):
        """
        This method is called by the reporter thread whenever a new metric
        value is received. It returns a dictionary with all the information
        needed for making decisions (e.g., stop / continue task, update
        model, etc)
        - task_continues: Should task continue or stop/pause?
        - milestone_reached: True if rung level (or max_t) is hit
        - next_milestone: If hit rung level < max_t, this is the subsequent
          rung level (otherwise: None)
        - bracket_id: Bracket in which the task is running

        :param task: Only task.task_id is used
        :param result: Current reported results from task
        :return: See above
        """
        rung_sys, bracket_id, skip_rungs = self._get_rung_system(task.task_id)
        ret_dict = {
            'bracket_id': bracket_id,
            'task_continues': False,
            'milestone_reached': True,
            'next_milestone': None
        }
        if self._scheduler_type == 'promotion':
            ret_dict['ignore_data'] = False
        if result[self._time_attr] < self._max_t:
            ret_dict.update(rung_sys.on_task_report(
                task, result[self._time_attr], result[self._reward_attr],
                skip_rungs=skip_rungs))
            # Special case: If config just reached the last milestone in
            # the bracket and survived, next_milestone is equal to max_t
            if ret_dict['task_continues'] and ret_dict['milestone_reached'] \
                    and (ret_dict['next_milestone'] is None):
                ret_dict['next_milestone'] = self._max_t
        return ret_dict

    def on_task_remove(self, task):
        task_id = task.task_id
        rung_sys, _, _ = self._get_rung_system(task_id)
        rung_sys.on_task_remove(task)
        del self._task_info[str(task_id)]

    def _sample_bracket(self):
        return _sample_bracket(
            num_brackets=self.num_brackets,
            rung_levels=self.rung_levels, random_state=self.random_state)

    def on_task_schedule(self):
        # Sample bracket for task to be scheduled
        bracket_id = self._sample_bracket()
        rung_sys, skip_rungs = self._get_rung_system_for_bracket_id(bracket_id)
        extra_kwargs = {'bracket': bracket_id}
        # Check whether config can be promoted
        ret_dict = rung_sys.on_task_schedule()
        config = ret_dict.get('config')
        if config is not None:
            extra_kwargs['milestone'] = ret_dict['next_milestone']
            extra_kwargs['config_key'] = ret_dict['config_key']
            extra_kwargs['resume_from'] = ret_dict['milestone']
        else:
            # First milestone the new config will get to
            extra_kwargs['milestone'] = rung_sys.get_first_milestone(
                skip_rungs)
        return config, extra_kwargs

    def snapshot_rungs(self, bracket_id):
        rung_sys, skip_rungs = self._get_rung_system_for_bracket_id(bracket_id)
        return rung_sys.snapshot_rungs(skip_rungs)

    def __repr__(self):
        reprstr = self.__class__.__name__ + '(' + \
                  'reward_attr: ' + self._reward_attr + \
                  ', time_attr: ' + self._time_attr + \
                  ', rung_levels: ' + str(self.rung_levels) + \
                  ', max_t: ' + str(self._max_t) + \
                  ', rung_systems: ' + str(self._rung_systems) + \
                  ')'
        return reprstr
