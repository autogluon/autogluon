import pickle
import logging
import threading
import numpy as np
import multiprocessing as mp
import os

from .fifo import FIFOScheduler
from .hyperband_stopping import HyperbandStopping_Manager
from .hyperband_promotion import HyperbandPromotion_Manager
from .reporter import DistStatusReporter
from ..utils import load

__all__ = ['HyperbandScheduler']

logger = logging.getLogger(__name__)


class HyperbandScheduler(FIFOScheduler):
    r"""Implements different variants of asynchronous Hyperband

    See 'type' for the different variants. One implementation detail is when
    using multiple brackets, task allocation to bracket is done randomly 
    based on a softmax probability.

    Note: This scheduler requires both reward and resource (time) to be
    returned by the reporter. Here, resource (time) values must be positive
    int. If time_attr == 'epoch', this should be the number of epochs done,
    starting from 1 (not the epoch number, starting from 0).

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
    reward_attr : str
        Name of reward (i.e., metric to maximize) attribute in data obtained
        from reporter
    time_attr : str
        Name of resource (or time) attribute in data obtained from reporter.
        Note: The type of resource must be positive int.
    max_t : int
        Maximum resource (see time_attr) to be used for a job. Together with
        grace_period and reduction_factor, this is used to determine rung
        levels in Hyperband brackets.
    grace_period : int
        Minimum resource (see time_attr) to be used for a job.
    reduction_factor : int (>= 2)
        Parameter to determine rung levels in successive halving (Hyperband).
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
                See :class:`HyperbandStopping_Manager`. Tasks and config evals are
                tightly coupled. A task is stopped at a milestone if worse than
                most others, otherwise it continues. As implemented in Ray/Tune:
                https://ray.readthedocs.io/en/latest/tune-schedulers.html#asynchronous-hyperband
            promotion:
                See :class:`HyperbandPromotion_Manager`. A config eval may be
                associated with multiple tasks over its lifetime. It is never
                terminated, but may be paused. Whenever a task becomes available,
                it may promote a config to the next milestone, if better than most
                others. If no config can be promoted, a new one is chosen. This
                variant may benefit from pause&resume, which is not directly
                supported here. As proposed in this paper (termed ASHA):
                https://arxiv.org/abs/1810.05934
    dist_ip_addrs : list of str
        IP addresses of remote machines.
    keep_size_ratios : bool
        Implemented for type 'promotion' only. If True,
        promotions are done only if the (current estimate of the) size ratio
        between rung and next rung are 1 / reduction_factor or better. This
        avoids higher rungs to get more populated than they would be in
        synchronous Hyperband. A drawback is that promotions to higher rungs
        take longer.
    maxt_pending : bool
        Relevant only if a model-based searcher is used.
        If True, register pending config at level max_t
        whenever a new evaluation is started. This has a direct effect on
        the acquisition function (for model-based variant), which operates
        at level max_t. On the other hand, it decreases the variance of the
        latent process there.
    searcher_data : str
        Relevant only if a model-based searcher is used, and if train_fn is such
        that we receive results (from the reporter) at each successive resource
        level, not just at the rung levels.
        Example: For NN tuning and time_attr == 'epoch', we receive a result for
        each epoch, but not all epoch values are also rung levels.
        searcher_data determines which of these results are passed to the
        searcher. As a rule, the more data the searcher receives, the better its
        fit, but also the more expensive get_config may become. Choices:
        - 'rungs' (default): Only results at rung levels. Cheapest
        - 'all': All results. Most expensive
        - 'rungs_and_last': Results at rung levels, plus the most recent result.
            This means that in between rung levels, only the most recent result
            is used by the searcher. This is in between

    See Also
    --------
    HyperbandStopping_Manager
    HyperbandPromotion_Manager


    Examples
    --------
    >>> import numpy as np
    >>> import autogluon as ag
    >>> @ag.args(
    ...     lr=ag.space.Real(1e-3, 1e-2, log=True),
    ...     wd=ag.space.Real(1e-3, 1e-2))
    >>> def train_fn(args, reporter):
    ...     print('lr: {}, wd: {}'.format(args.lr, args.wd))
    ...     for e in range(10):
    ...         dummy_accuracy = 1 - np.power(1.8, -np.random.uniform(e, 2*e))
    ...         reporter(epoch=e+1, accuracy=dummy_accuracy, lr=args.lr, wd=args.wd)
    >>> scheduler = ag.scheduler.HyperbandScheduler(train_fn,
    ...                                             resource={'num_cpus': 2, 'num_gpus': 0},
    ...                                             num_trials=20,
    ...                                             reward_attr='accuracy',
    ...                                             time_attr='epoch',
    ...                                             grace_period=1)
    >>> scheduler.run()
    >>> scheduler.join_jobs()
    >>> scheduler.get_training_curves(plot=True)
    """
    def __init__(self, train_fn, args=None, resource=None,
                 searcher=None, search_options=None,
                 checkpoint=None,
                 resume=False, num_trials=None,
                 time_out=None, max_reward=None,
                 reward_attr="accuracy",
                 time_attr="epoch",
                 max_t=50, grace_period=1,
                 reduction_factor=3, brackets=1,
                 visualizer='none',
                 training_history_callback=None,
                 training_history_callback_delta_secs=60,
                 delay_get_config=True,
                 type='stopping',
                 dist_ip_addrs=None,
                 keep_size_ratios=False,
                 maxt_pending=False,
                 searcher_data='rungs',
                 do_snapshots=False):
        # Adjoin information about scheduler to search_options
        if search_options is None:
            _search_options = dict()
        else:
            _search_options = search_options.copy()
        _search_options['scheduler'] = 'hyperband_{}'.format(type)
        _search_options['resource_attribute'] = time_attr
        _search_options['min_epochs'] = grace_period
        _search_options['max_epochs'] = max_t
        # Pass resume=False here. Resume needs members of this object to be
        # created
        super().__init__(
            train_fn=train_fn, args=args, resource=resource, searcher=searcher,
            search_options=_search_options, checkpoint=checkpoint, resume=False,
            num_trials=num_trials, time_out=time_out, max_reward=max_reward,
            reward_attr=reward_attr, time_attr=time_attr,
            visualizer=visualizer, dist_ip_addrs=dist_ip_addrs,
            training_history_callback=training_history_callback,
            training_history_callback_delta_secs=training_history_callback_delta_secs,
            delay_get_config=delay_get_config)
        self.max_t = max_t
        self.reduction_factor = reduction_factor
        self.type = type
        self.maxt_pending = maxt_pending
        if type == 'stopping':
            self.terminator = HyperbandStopping_Manager(
                time_attr, reward_attr, max_t, grace_period, reduction_factor,
                brackets)
        elif type == 'promotion':
            assert not do_snapshots, \
                "Snapshots are supported only for type = 'stopping'"
            self.terminator = HyperbandPromotion_Manager(
                time_attr, reward_attr, max_t, grace_period, reduction_factor,
                brackets, keep_size_ratios=keep_size_ratios)
        else:
            raise AssertionError(
                "type '{}' not supported, must be 'stopping' or 'promotion'".format(
                    type))
        self.do_snapshots = do_snapshots
        sd_values = {'rungs', 'all', 'rungs_and_last'}
        assert searcher_data in sd_values, \
            "searcher_data = '{}', must be in {}".format(
                searcher_data, sd_values)
        self.searcher_data = searcher_data
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
        # - keep_case: See _run_reporter
        self._running_tasks = dict()
        # This lock protects both _running_tasks and terminator, the latter
        # does not define its own lock
        self._hyperband_lock = mp.Lock()
        if resume:
            assert checkpoint is not None, \
                "Need checkpoint to be set if resume = True"
            if os.path.isfile(checkpoint):
                self.load_state_dict(load(checkpoint))
            else:
                msg = f'checkpoint path {checkpoint} is not available for resume.'
                logger.exception(msg)
                raise FileExistsError(msg)

    def add_job(self, task, **kwargs):
        """Adding a training task to the scheduler.

        Args:
            task (:class:`autogluon.scheduler.Task`): a new training task

        Relevant entries in kwargs:
        - bracket: HB bracket to be used. Has been sampled in _promote_config
        - new_config: If True, task starts new config eval, otherwise it promotes
          a config (only if type == 'promotion')
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
        reporter = DistStatusReporter(remote=task.resources.node)
        task.args['reporter'] = reporter

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
            milestones = self.terminator.on_task_add(task, **kwargs)
        if kwargs.get('new_config', True):
            first_milestone = milestones[-1]
            logger.debug("Adding new task (first milestone = {}):\n{}".format(
                first_milestone, task))
            self.searcher.register_pending(
                task.args['config'], milestone=first_milestone)
            if self.maxt_pending:
                # Also introduce pending evaluation for resource max_t
                final_milestone = milestones[0]
                if final_milestone != first_milestone:
                    self.searcher.register_pending(
                        task.args['config'], milestone=final_milestone)
        else:
            # Promotion of config
            # This is a signal towards train_fn, in case it supports
            # pause and resume:
            task.args['resume_from'] = kwargs['resume_from']
            next_milestone = kwargs['milestone']
            logger.debug("Promotion task (next milestone = {}):\n{}".format(
                next_milestone, task))
            self.searcher.register_pending(
                task.args['config'], milestone=next_milestone)

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

    def _update_searcher(self, task, result):
        config = task.args['config']
        if self.searcher_data == 'rungs_and_last':
            with self._hyperband_lock:
                task_info = self._running_tasks[str(task.task_id)]
                if task_info['reported_result'] is not None:
                    # Remove last recently added result for this task,
                    # unless it fell on a rung level
                    if not task_info['keep_case']:
                        rem_result = task_info['reported_result']
                        self.searcher.remove_case(config, **rem_result)
        self.searcher.update(config, **result)

    def _run_reporter(self, task, task_job, reporter):
        last_result = None
        last_updated = None
        task_key = str(task.task_id)
        while not task_job.done():
            reported_result = reporter.fetch()
            if 'traceback' in reported_result:
                # Evaluation has failed
                logger.exception(reported_result['traceback'])
                self.searcher.evaluation_failed(
                    config=task.args['config'], **reported_result)
                reporter.move_on()
                with self._hyperband_lock:
                    self.terminator.on_task_remove(task)
                break
            if reported_result.get('done', False):
                reporter.move_on()
                with self._hyperband_lock:
                    if last_result is not None:
                        self.terminator.on_task_complete(task, last_result)
                    # Cleanup
                    del self._running_tasks[task_key]
                break
            if len(reported_result) == 0:
                # An empty dict should just be skipped
                if self.searcher.debug_log is not None:
                    logger.info("Skipping empty dict received from reporter")
                continue
            # Time since start of experiment
            elapsed_time = self._elapsed_time()
            reported_result['time_since_start'] = elapsed_time

            # Call before _add_training_results, since we may be able to report
            # extra information from the bracket
            with self._hyperband_lock:
                task_info = self.terminator.on_task_report(
                    task, reported_result)
            task_continues = task_info['task_continues']
            update_searcher = task_info['update_searcher']
            # Append extra information to reported_result
            reported_result['bracket'] = task_info['bracket_id']
            if 'rung_counts' in task_info:
                for k, v in task_info['rung_counts'].items():
                    key = 'count_at_{}'.format(k)
                    reported_result[key] = v
            dataset_size = self.searcher.dataset_size()
            if dataset_size > 0:
                reported_result['searcher_data_size'] = dataset_size
            for k, v in self.searcher.cumulative_profile_record().items():
                reported_result['searcher_profile_' + k] = v
            for k, v in self.searcher.model_parameters().items():
                reported_result['searcher_params_' + k] = v
            self._add_training_result(
                task.task_id, reported_result, config=task.args['config'])

            if self.searcher_data == 'rungs':
                # Only results on rung levels are reported to the searcher
                if task_continues and update_searcher:
                    # Update searcher with intermediate result
                    # Note: If task_continues is False here, we also call
                    # searcher.update, but outside the loop.
                    self._update_searcher(task, reported_result)
                    last_updated = reported_result
                    next_milestone = task_info.get('next_milestone')
                    if next_milestone is not None:
                        self.searcher.register_pending(
                            task.args['config'], milestone=next_milestone)
            elif not task_info.get('ignore_data', False):
                # All results are reported to the searcher
                # When is task_info['ignore_data'] True? See header comment of
                # HyperbandPromotion_Manager.
                self._update_searcher(task, reported_result)
                last_updated = reported_result
                # Since all results are reported, the next report for this task
                # will be for resource + 1.
                # NOTE: This assumes that results are reported for all successive
                # resource levels (int). If any resource level is skipped,
                # there may be left-over pending candidates, who will be
                # removed once the task finishes.
                if task_continues:
                    self.searcher.register_pending(
                        task.args['config'],
                        milestone=int(reported_result[self._time_attr]) + 1)

            # Change snapshot entry for task
            # Note: This must not be done above, because what _update_searcher
            # is doing, depends on the entry *before* its update here.
            # If searcher_data == 'rungs_and_last', the result is kept in
            # the dataset iff update_searcher == True (i.e., we are at a
            # rung level).
            with self._hyperband_lock:
                # Note: reported_result may contain all sorts of extra info.
                # All we need to maintain in the snapshot are reward and
                # resource level
                self._running_tasks[task_key].update({
                    'time_stamp': elapsed_time,
                    'reported_result': {
                        self._reward_attr: reported_result[self._reward_attr],
                        self._time_attr: reported_result[self._time_attr]},
                    'keep_case': update_searcher})

            last_result = reported_result
            if task_continues:
                debug_log = self.searcher.debug_log
                if update_searcher and debug_log is not None:
                    # Debug log output
                    config_id = debug_log.config_id(task.args['config'])
                    milestone = int(reported_result[self._time_attr])
                    next_milestone = task_info['next_milestone']
                    msg = "config_id {}: Reaches {}, continues".format(
                        config_id, milestone)
                    if next_milestone is not None:
                        msg += " to {}".format(next_milestone)
                    logger.info(msg)
                reporter.move_on()
            else:
                # Note: The 'terminated' signal is sent even in the promotion
                # variant. It means that the *task* terminates, while the
                # evaluation of the config is just paused
                last_result['terminated'] = True
                debug_log = self.searcher.debug_log
                if debug_log is not None:
                    # Debug log output
                    config_id = debug_log.config_id(task.args['config'])
                    resource = int(reported_result[self._time_attr])
                    if self.type == 'stopping' or resource >= self.max_t:
                        act_str = 'Terminating'
                    else:
                        act_str = 'Pausing'
                    msg = "config_id {}: {} evaluation at {}".format(
                        config_id, act_str, resource)
                    logger.info(msg)
                with self._hyperband_lock:
                    self.terminator.on_task_remove(task)
                    # Cleanup
                    del self._running_tasks[task_key]
                reporter.terminate()
                break

        # Pass all of last_result to searcher (unless this has already been
        # done)
        if last_result is not last_updated:
            self._update_searcher(task, last_result)

    def state_dict(self, destination=None):
        """Returns a dictionary containing a whole state of the Scheduler

        Examples
        --------
        >>> ag.save(scheduler.state_dict(), 'checkpoint.ag')
        """
        destination = super().state_dict(destination)
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

    def _snapshot_tasks(self, bracket_id):
        return {
            k: {'config': v['config'],
                'time': v['time_stamp'],
                'level': 0 if v['reported_result'] is None
                else v['reported_result'][self._time_attr]}
            for k, v in self._running_tasks.items()
            if v['bracket'] == bracket_id}

    # Snapshot (in extra_kwargs['snapshot']):
    # - max_resource
    # - reduction_factor
    # - tasks: Info about running tasks in bracket bracket_id:
    #   dict(task_id) -> dict:
    #   - config: config as dict
    #   - time: Time when task was started, or when last recent result was
    #     reported
    #   - level: Level of last recent result report, or 0 if no reports yet
    # - rungs: Metric values at rung levels in bracket bracket_id:
    #   List of (rung_level, metric_dict), where metric_dict has entries
    #   task_id: metric_value.
    def _promote_config(self):
        with self._hyperband_lock:
            config, extra_kwargs = self.terminator.on_task_schedule()
            if self.do_snapshots:
                # Append snapshot
                bracket_id = extra_kwargs['bracket']
                extra_kwargs['snapshot'] = {
                    'tasks': self._snapshot_tasks(bracket_id),
                    'rungs': self.terminator.snapshot_rungs(bracket_id),
                    'max_resource': self.max_t,
                    'reduction_factor': self.reduction_factor}
            debug_log = self.searcher.debug_log
            if (debug_log is not None) and (config is not None):
                # Debug log output
                config_id = debug_log.config_id(config)
                msg = "config_id {}: Promotion from {} to {}".format(
                    config_id, extra_kwargs['resume_from'],
                    extra_kwargs['milestone'])
                logger.info(msg)
            return config, extra_kwargs

    def map_resource_to_index(self):
        def fun(resource):
            assert 0.0 <= resource <= 1.0, \
                "resource must be in [0, 1]"
            return self.terminator.resource_to_index(resource)

        return fun

    def __repr__(self):
        reprstr = self.__class__.__name__ + '(' +  \
            'terminator: ' + str(self.terminator)
        return reprstr
