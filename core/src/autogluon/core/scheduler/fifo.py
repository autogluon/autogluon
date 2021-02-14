import copy
import json
import logging
import multiprocessing as mp
import os
import pickle
import threading
import time
from collections import OrderedDict
from time import sleep

import numpy as np
from tqdm.auto import tqdm

from .reporter import DistStatusReporter, FakeReporter
from .resource import DistributedResource
from .scheduler import TaskScheduler
from ..decorator import _autogluon_method
from ..searcher import BaseSearcher
from ..searcher import searcher_factory
from ..task.task import Task
from ..utils import save, load, mkdir, try_import_mxboard
from ..utils.default_arguments import check_and_merge_defaults, \
    Float, Integer, String, Boolean, assert_no_invalid_options

__all__ = ['FIFOScheduler']

logger = logging.getLogger(__name__)


_ARGUMENT_KEYS = {
    'args', 'resource', 'searcher', 'search_options', 'checkpoint', 'resume',
    'num_trials', 'time_out', 'max_reward', 'reward_attr', 'time_attr',
    'elapsed_time_attr', 'dist_ip_addrs', 'visualizer',
    'training_history_callback', 'training_history_callback_delta_secs',
    'training_history_searcher_info', 'delay_get_config'}

_DEFAULT_OPTIONS = {
    'resource': {'num_cpus': 1, 'num_gpus': 0},
    'searcher': 'random',
    'resume': False,
    'reward_attr': 'accuracy',
    'time_attr': 'epoch',
    'elapsed_time_attr': 'elapsed_time',
    'visualizer': 'none',
    'training_history_callback_delta_secs': 60,
    'training_history_searcher_info': False,
    'delay_get_config': True}

_CONSTRAINTS = {
    'checkpoint': String(),
    'resume': Boolean(),
    'num_trials': Integer(1, None),
    'time_out': Float(0.0, None),
    'max_reward': Float(),
    'reward_attr': String(),
    'time_attr': String(),
    'elapsed_time_attr': String(),
    'visualizer': String(),
    'training_history_callback_delta_secs': Integer(1, None),
    'training_history_searcher_info': Boolean(),
    'delay_get_config': Boolean()}


class FIFOScheduler(TaskScheduler):
    r"""Simple scheduler that just runs trials in submission order.

    Parameters
    ----------
    train_fn : callable
        A task launch function for training.
    args : object (optional)
        Default arguments for launching train_fn.
    resource : dict
        Computation resources. For example, `{'num_cpus':2, 'num_gpus':1}`
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
        This attribute is optional for FIFO scheduling, but becomes mandatory
        in multi-fidelity scheduling (e.g., Hyperband).
        Note: The type of resource must be int.
    elapsed_time_attr : str (optional)
        Needed only with searchers who model train_fn evaluation times. Name
        of elapsed time attribute in data obtained from reporter. Here,
        elapsed time counts since the start of train_fn, unit is seconds. Type
        must be positive float.
    dist_ip_addrs : list of str
        IP addresses of remote machines.
    training_history_callback : callable
        Callback function called every time a result is added to
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


    Examples
    --------
    >>> import numpy as np
    >>> import autogluon.core as ag
    >>> @ag.args(
    ...     lr=ag.space.Real(1e-3, 1e-2, log=True),
    ...     wd=ag.space.Real(1e-3, 1e-2))
    >>> def train_fn(args, reporter):
    ...     print('lr: {}, wd: {}'.format(args.lr, args.wd))
    ...     for e in range(10):
    ...         dummy_accuracy = 1 - np.power(1.8, -np.random.uniform(e, 2*e))
    ...         reporter(epoch=e+1, accuracy=dummy_accuracy, lr=args.lr, wd=args.wd)
    >>> scheduler = ag.scheduler.FIFOScheduler(train_fn,
    ...                                        resource={'num_cpus': 2, 'num_gpus': 0},
    ...                                        num_trials=20,
    ...                                        reward_attr='accuracy',
    ...                                        time_attr='epoch')
    >>> scheduler.run()
    >>> scheduler.join_jobs()
    >>> scheduler.get_training_curves(plot=True)
    """

    def __init__(self, train_fn, **kwargs):
        super().__init__(kwargs.get('dist_ip_addrs'))
        # Check values and impute default values
        assert_no_invalid_options(
            kwargs, _ARGUMENT_KEYS, name='FIFOScheduler')
        kwargs = check_and_merge_defaults(
            kwargs, set(), _DEFAULT_OPTIONS, _CONSTRAINTS,
            dict_name='scheduler_options')

        self.resource = kwargs['resource']
        searcher = kwargs['searcher']
        search_options = kwargs.get('search_options')
        if isinstance(searcher, str):
            if search_options is None:
                search_options = dict()
            _search_options = search_options.copy()
            _search_options['configspace'] = train_fn.cs
            _search_options['reward_attribute'] = kwargs['reward_attr']
            _search_options['resource_attribute'] = kwargs['time_attr']
            _search_options['elapsed_time_attribute'] = \
                kwargs['elapsed_time_attr']
            # Adjoin scheduler info to search_options, if not already done by
            # subclass
            if 'scheduler' not in _search_options:
                _search_options['scheduler'] = 'fifo'
            self.searcher: BaseSearcher = searcher_factory(
                searcher, **_search_options)
        else:
            assert isinstance(searcher, BaseSearcher)
            self.searcher: BaseSearcher = searcher

        assert isinstance(train_fn, _autogluon_method)
        self.train_fn = train_fn
        args = kwargs.get('args')
        self.args = args if args else train_fn.args
        num_trials = kwargs.get('num_trials')
        time_out = kwargs.get('time_out')
        if num_trials is None:
            assert time_out is not None, \
                "Need stopping criterion: Either num_trials or time_out"
        self.num_trials = num_trials
        self.time_out = time_out
        self.max_reward = kwargs.get('max_reward')
        # meta data
        self.metadata = {
            'search_space': train_fn.kwspaces,
            'search_strategy': searcher,
            'stop_criterion': {
                'time_limits': time_out,
                'max_reward': self.max_reward},
            'resources_per_trial': self.resource}

        checkpoint = kwargs.get('checkpoint')
        self._checkpoint = checkpoint
        self._reward_attr = kwargs['reward_attr']
        self._time_attr = kwargs['time_attr']
        self._elapsed_time_attr = kwargs['elapsed_time_attr']
        self.visualizer = kwargs['visualizer'].lower()
        if self.visualizer == 'tensorboard' or self.visualizer == 'mxboard':
            assert checkpoint is not None, "Need checkpoint to be set"
            try_import_mxboard()
            from mxboard import SummaryWriter
            self.mxboard = SummaryWriter(
                logdir=os.path.join(os.path.splitext(checkpoint)[0], 'logs'),
                flush_secs=3,
                verbose=False
            )

        self._fifo_lock = mp.Lock()
        # training_history maintains the complete history of the experiment,
        # in terms of all results obtained from the reporter. Keys are
        # str(task.task_id)
        self.training_history = OrderedDict()
        self.config_history = OrderedDict()
        # Needed for training_history callback mechanism, which is used to
        # serialize training_history after each
        # training_history_call_delta_secs seconds
        self._start_time = None
        self._training_history_callback_last_block = None
        self.training_history_callback = kwargs.get('training_history_callback')
        self.training_history_callback_delta_secs = \
            kwargs['training_history_callback_delta_secs']
        self.training_history_searcher_info = \
            kwargs['training_history_searcher_info']
        self._delay_get_config = kwargs['delay_get_config']
        # Resume experiment from checkpoint?
        if kwargs['resume']:
            assert checkpoint is not None, \
                "Need checkpoint to be set if resume = True"
            if os.path.isfile(checkpoint):
                self.load_state_dict(load(checkpoint))
            else:
                msg = f'checkpoint path {checkpoint} is not available for resume.'
                logger.critical(msg)
                raise FileExistsError(msg)

    def run(self, **kwargs):
        """Run multiple number of trials
        """
        # Make sure that this scheduler is configured at the searcher
        self.searcher.configure_scheduler(self)
        start_time = time.time()
        self._start_time = start_time
        num_trials = kwargs.get('num_trials', self.num_trials)
        time_out = kwargs.get('time_out', self.time_out)
        # For training_history callback mechanism:
        self._training_history_callback_last_block = -1
        log_suffix = ''
        if time_out is not None:
            log_suffix = f' (time_out={round(time_out, 2)}s)'
        elif num_trials is not None:
            log_suffix = f' (num_trials={num_trials - self.num_finished_tasks})'
        logger.info(f'Starting Hyperparameter Tuning ...{log_suffix}')
        logger.log(15, f'Num of Finished Tasks is {self.num_finished_tasks}')
        if num_trials is not None:
            logger.log(15, f'Num of Pending Tasks is {num_trials - self.num_finished_tasks}')
            tbar = tqdm(range(self.num_finished_tasks, num_trials))
        else:
            # In this case, only stopping by time_out is used. We do not display
            # a progress bar then
            tbar = range(self.num_finished_tasks, 100000)
        for _ in tbar:
            # Quick check if resources are available before we check the time limit
            # This is to prevent booking next job while we are waiting for a resource, which
            # results in going over the time limit
            resources = DistributedResource(**self.resource)
            while not FIFOScheduler.managers.check_availability(resources):
                sleep(0.1)

            if (time_out and time.time() - start_time >= time_out) or \
                    (self.max_reward and self.get_best_reward() >= self.max_reward):
                break
            self.schedule_next()

    def save(self, checkpoint=None):
        """Save Checkpoint
        """
        if checkpoint is None:
            checkpoint = self._checkpoint
        if checkpoint is not None:
            mkdir(os.path.dirname(checkpoint))
            save(self.state_dict(), checkpoint)

    def _create_new_task(self, config, resources=None):
        if resources is None:
            resources = DistributedResource(**self.resource)
        return Task(
            self.train_fn, {'args': self.args, 'config': config},
            resources=resources)

    def schedule_next(self):
        """Schedule next searcher suggested task
        """
        resources = DistributedResource(**self.resource)
        if self._delay_get_config:
            # Wait for available resource here, instead of in add_job. This
            # delays the get_config call until a resource is available
            FIFOScheduler.managers.request_resources(resources)

        # Allow for the promotion of a previously chosen config. Also,
        # extra_kwargs contains extra info passed to both add_job and to
        # get_config (if no config is promoted)
        config, extra_kwargs = self._promote_config()
        # Time stamp to be used in get_config, and maybe in add_job
        extra_kwargs['elapsed_time'] = self._elapsed_time()
        if config is None:
            # No config to promote: Query next config to evaluate from searcher
            config = self.searcher.get_config(**extra_kwargs)
            extra_kwargs['new_config'] = True
        else:
            # This is not a new config, but a paused one which is now promoted
            extra_kwargs['new_config'] = False
        task = self._create_new_task(config, resources=resources)
        self.add_job(task, **extra_kwargs)

    def run_with_config(self, config):
        """Run with config for final fit.
        It launches a single training trial under any fixed values of the hyperparameters.
        For example, after HPO has identified the best hyperparameter values based on a hold-out dataset,
        one can use this function to retrain a model with the same hyperparameters on all the available labeled data
        (including the hold out set). It can also returns other objects or states.
        """
        task = self._create_new_task(config)
        reporter = FakeReporter()
        task.args['reporter'] = reporter
        return self.run_job(task)

    def _dict_from_task(self, task):
        if isinstance(task, Task):
            return {'TASK_ID': task.task_id, 'Config': task.args['config']}
        else:
            assert isinstance(task, dict)
            return {'TASK_ID': task['TASK_ID'], 'Config': task['Config']}

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
        cls = FIFOScheduler
        if not self._delay_get_config:
            # Wait for resource to become available here, as this has not happened
            # in schedule_next before
            cls.managers.request_resources(task.resources)
        # reporter
        reporter = DistStatusReporter(remote=task.resources.node)
        task.args['reporter'] = reporter
        # Register pending evaluation
        self.searcher.register_pending(task.args['config'])
        # main process
        debug_log = self.searcher.debug_log
        if debug_log is not None:
            logger.info("Starting job on node = {}".format(
                task.resources.node.remote_id))
            # Relay config_id to train_fn (for debug logging)
            config_id = debug_log.config_id(task.args['config'])
            task.args['args']['config_id'] = config_id
        job = cls.jobs.start_distributed_job(task, cls.managers)
        # reporter thread
        rp = threading.Thread(
            target=self._run_reporter,
            args=(task, job, reporter),
            daemon=False)
        rp.start()
        task_dict = self._dict_from_task(task)
        task_dict.update({'Task': task, 'Job': job, 'ReporterThread': rp})
        # Checkpoint thread
        if self._checkpoint is not None:
            self._add_checkpointing_to_job(job)
        with self.managers.lock:
            self.scheduled_tasks.append(task_dict)

    def _clean_task_internal(self, task_dict):
        task_dict['ReporterThread'].join()

    def _add_checkpointing_to_job(self, job):
        def _save_checkpoint_callback(fut):
            self._cleaning_tasks()
            self.save()

        job.add_done_callback(_save_checkpoint_callback)

    def _training_history_callback_current_block(self):
        return int(np.floor(
            self._elapsed_time() / self.training_history_callback_delta_secs))

    def _trigger_training_history_callback(self):
        """
        Decides whether 'training_history_callback' is to be called. This
        callback is (in general) serializing 'training_history' (or required
        parts of it). It may also serialize the current scheduler state.

        To this end, we partition time since start of experiment (see
        `_elapsed_time') into blocks of 'training_history_callback_delta_secs'
        seconds. The callback is triggered only if the last recent call
        happened in an earlier block.

        :return: Should callback be executed?
        """
        if self.training_history_callback is None:
            return False
        assert self._training_history_callback_last_block is not None
        current_block = self._training_history_callback_current_block()
        return (current_block > self._training_history_callback_last_block)

    def _run_reporter(self, task, task_job, reporter):
        last_result = None
        while not task_job.done():
            reported_result = reporter.fetch()
            if 'traceback' in reported_result:
                # Evaluation has failed
                logger.critical(reported_result['traceback'])
                self.searcher.evaluation_failed(
                    config=task.args['config'], **reported_result)
                reporter.move_on()
                break
            if reported_result.get('done', False):
                reporter.move_on()
                break
            if len(reported_result) == 0:
                # An empty dict should just be skipped
                logger.warning("Skipping empty dict received from reporter")
                continue

            # Extra information from searcher (optional)
            if self.training_history_searcher_info:
                dataset_size = self.searcher.dataset_size()
                if dataset_size > 0:
                    reported_result['searcher_data_size'] = dataset_size
                for k, v in self.searcher.model_parameters().items():
                    reported_result['searcher_params_' + k] = v
            self._add_training_result(
                task.task_id, reported_result, config=task.args['config'])
            reporter.move_on()
            last_result = reported_result
        # Pass all of last_result to searcher
        if last_result is not None:
            self.searcher.update(config=task.args['config'], **last_result)

    def _promote_config(self):
        """
        Provides a hook in schedule_next, which allows to promote a config
        which has been selected and partially evaluated previously.

        :return: config, extra_args
        """
        config = None
        extra_args = dict()
        return config, extra_args

    def _elapsed_time(self):
        """
        :return: Time elapsed since start of experiment (see 'run')
        """
        assert self._start_time is not None, \
            "Experiment has not been started yet"
        return time.time() - self._start_time

    def get_best_config(self):
        """Get the best configuration from the finished jobs.
        """
        return self.searcher.get_best_config()

    def get_best_task_id(self):
        """Get the task id that results in the best configuration/best reward.

        If there are duplicated configurations, we return the id of the first one.
        """
        best_config = self.get_best_config()
        with self._fifo_lock:
            for task_id, config in self.config_history.items():
                if pickle.dumps(best_config) == pickle.dumps(config):
                    return task_id
            raise RuntimeError('The best config {} is not found in config history = {}. '
                               'This should never happen!'.format(best_config, self.config_history))

    def get_best_reward(self):
        """Get the best reward from the finished jobs.
        """
        return self.searcher.get_best_reward()

    def _add_training_result(self, task_id, reported_result, config=None):
        if self.visualizer == 'mxboard' or self.visualizer == 'tensorboard':
            if 'loss' in reported_result:
                self.mxboard.add_scalar(
                    tag='loss',
                    value=(
                        f'task {task_id} valid_loss',
                        reported_result['loss']
                    ),
                    global_step=reported_result[self._time_attr]
                )
            self.mxboard.add_scalar(
                tag=self._reward_attr,
                value=(
                    f'task {task_id} {self._reward_attr}',
                    reported_result[self._reward_attr]
                ),
                global_step=reported_result[self._time_attr]
            )
        # This locking block includes the (optional) call of
        # training_history_callback. We need to make sure that it is not called
        # by two reporter threads at the same time
        with self._fifo_lock:
            # Note: We store all of reported_result in
            # training_history[task_id], not just the reward value.
            task_key = str(task_id)
            new_entry = copy.copy(reported_result)
            if task_key in self.training_history:
                self.training_history[task_key].append(new_entry)
            else:
                self.training_history[task_key] = [new_entry]
                if config:
                    self.config_history[task_key] = config
            # training_history_callback mechanism
            if self._trigger_training_history_callback():
                # Note: no_fifo_lock = True avoids deadlock in 'state_dict',
                # since we acquired the _fifo_lock already
                self.training_history_callback(
                    self.training_history,
                    self._start_time,
                    config_history=self.config_history,
                    state_dict=self.state_dict(no_fifo_lock=True))
                self._training_history_callback_last_block = \
                    self._training_history_callback_current_block()


    def get_training_curves(self, filename=None, plot=False, use_legend=True):
        """Get Training Curves

        Parameters
        ----------
            filename : str
            plot : bool
            use_legend : bool

        Examples
        --------
        >>> scheduler.run()
        >>> scheduler.join_jobs()
        >>> scheduler.get_training_curves(plot=True)

            .. image:: https://github.com/zhanghang1989/AutoGluonWebdata/blob/master/doc/api/autogluon.1.png?raw=true
        """
        if filename is None and not plot:
            logger.warning('Please either provide filename or allow plot in get_training_curves')
        import matplotlib.pyplot as plt

        eval_metric = self.__get_training_history_metric('eval_metric', default='validation_performance')
        sign_mult = int(self.__get_training_history_metric('greater_is_better', default=True)) * 2 - 1

        plt.ylabel(eval_metric)
        plt.xlabel(self._time_attr)
        plt.title("Performance vs Training-Time in each HPO Trial")
        with self._fifo_lock:
            for task_id, task_res in self.training_history.items():
                rewards = [x[self._reward_attr] * sign_mult for x in task_res]
                x = [x[self._time_attr] for x in task_res]
                plt.plot(x, rewards, label=f'task {task_id}')
        if use_legend:
            plt.legend(loc='best')
        if filename:
            logger.info(f'Saving Training Curve in {filename}')
            plt.savefig(filename)
        if plot:
            plt.show()

    def __get_training_history_metric(self, metric, default=None):
        for _, task_res in self.training_history.items():
            if task_res and metric in task_res[0]:
                return task_res[0][metric]
        return default

    def state_dict(self, destination=None, no_fifo_lock=False):
        """
        Returns a dictionary containing a whole state of the Scheduler. This is
        used for checkpointing.

        Note that the checkpoint only contains information which has been
        registered at scheduler and searcher. It does not contain information
        about currently running jobs, except what they reported before the
        checkpoint.
        Therefore, resuming an experiment from a checkpoint is slightly
        different from continuing the experiment past the checkpoint. The
        former behaves as if all currently running jobs are terminated at
        the checkpoint, and new jobs are scheduled from there, starting from
        scheduler and searcher state according to all information recorded
        until the checkpoint.

        Examples
        --------
        >>> ag.save(scheduler.state_dict(), 'checkpoint.ag')
        """
        destination = super().state_dict(destination)
        if not no_fifo_lock:
            self._fifo_lock.acquire()
        try:
            # The result of searcher.get_state can always be pickled
            destination['searcher'] = pickle.dumps(self.searcher.get_state())
            destination['training_history'] = json.dumps(self.training_history)
            destination['config_history'] = json.dumps(self.config_history)
        finally:
            if not no_fifo_lock:
                self._fifo_lock.release()
        if self.visualizer == 'mxboard' or self.visualizer == 'tensorboard':
            destination['visualizer'] = json.dumps(self.mxboard._scalar_dict)
        return destination

    def load_state_dict(self, state_dict):
        """
        Load from the saved state dict. This can be used to resume an
        experiment from a checkpoint (see 'state_dict' for caveats).

        This method must only be called as part of scheduler construction.
        Calling it in the middle of an experiment can lead to an undefined
        inner state of scheduler or searcher.

        Examples
        --------
        >>> scheduler.load_state_dict(ag.load('checkpoint.ag'))
        """
        super().load_state_dict(state_dict)
        with self._fifo_lock:
            self.searcher = self.searcher.clone_from_state(
                pickle.loads(state_dict['searcher']))
            self.training_history = json.loads(state_dict['training_history'])
            self.config_history = json.loads(state_dict['config_history'])
        if self.visualizer == 'mxboard' or self.visualizer == 'tensorboard':
            self.mxboard._scalar_dict = json.loads(state_dict['visualizer'])
        logger.debug(f'Loading Searcher State {self.searcher}')
