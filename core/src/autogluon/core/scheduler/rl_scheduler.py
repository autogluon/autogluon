import os
import json
import time
import pickle
import logging
import threading
import multiprocessing as mp
from collections import OrderedDict

from .fifo import FIFOScheduler
from .reporter import DistStatusReporter
from .resource import DistributedResource
from .. import Task
from ..decorator import _autogluon_method
from ..searcher import RLSearcher
from ..utils import load, tqdm, try_import_mxnet
from ..utils.default_arguments import check_and_merge_defaults, \
    Integer, Boolean, Float, String, filter_by_key

__all__ = ['RLScheduler']

logger = logging.getLogger(__name__)


_ARGUMENT_KEYS = {
    'controller_lr', 'ema_baseline_decay', 'controller_resource',
    'controller_batch_size', 'sync'}

_DEFAULT_OPTIONS = {
    'resume': False,
    'reward_attr': 'accuracy',
    'checkpoint': './exp/checkpoint.ag',
    'controller_lr': 1e-3,
    'ema_baseline_decay': 0.95,
    'controller_resource': {'num_cpus': 0, 'num_gpus': 0},
    'controller_batch_size': 1,
    'sync': True}

_CONSTRAINTS = {
    'resume': Boolean(),
    'reward_attr': String(),
    'checkpoint': String(),
    'controller_lr': Float(0.0, None),
    'ema_baseline_decay': Float(0.0, 1.0),
    'controller_batch_size': Integer(1, None),
    'sync': Boolean()}


class RLScheduler(FIFOScheduler):
    r"""Scheduler that uses Reinforcement Learning with a LSTM controller created based on the provided search spaces

    Parameters
    ----------
    train_fn : callable
        A task launch function for training. Note: please add the `@ag.args` decorater to the original function.
    args : object (optional)
        Default arguments for launching train_fn.
    resource : dict
        Computation resources.  For example, `{'num_cpus':2, 'num_gpus':1}`
    searcher : object (optional)
        Autogluon searcher.  For example, autogluon.searcher.RandomSearcher
    time_attr : str
        A training result attr to use for comparing time.
        Note that you can pass in something non-temporal such as
        `training_epoch` as a measure of progress, the only requirement
        is that the attribute should increase monotonically.
    reward_attr : str
        The training result objective value attribute. As with `time_attr`, this may refer to any objective value.
        Stopping procedures will use this attribute.
    controller_resource : int
        Batch size for training controllers.
    dist_ip_addrs : list of str
        IP addresses of remote machines.

    Examples
    --------
    >>> import numpy as np
    >>> import autogluon.core as ag
    >>>
    >>> @ag.args(
    ...     lr=ag.space.Real(1e-3, 1e-2, log=True),
    ...     wd=ag.space.Real(1e-3, 1e-2))
    >>> def train_fn(args, reporter):
    ...     print('lr: {}, wd: {}'.format(args.lr, args.wd))
    ...     for e in range(10):
    ...         dummy_accuracy = 1 - np.power(1.8, -np.random.uniform(e, 2*e))
    ...         reporter(epoch=e+1, accuracy=dummy_accuracy, lr=args.lr, wd=args.wd)
    ...
    >>> scheduler = ag.scheduler.RLScheduler(train_fn,
    ...                                      resource={'num_cpus': 2, 'num_gpus': 0},
    ...                                      num_trials=20,
    ...                                      reward_attr='accuracy',
    ...                                      time_attr='epoch')
    >>> scheduler.run()
    >>> scheduler.join_jobs()
    >>> scheduler.get_training_curves(plot=True)
    """
    def __init__(self, train_fn, **kwargs):
        try_import_mxnet()
        import mxnet as mx

        assert isinstance(train_fn, _autogluon_method), 'Please use @ag.args ' + \
                'to decorate your training script.'
        # Check values and impute default values (only for arguments new to
        # this class)
        kwargs = check_and_merge_defaults(
            kwargs, set(), _DEFAULT_OPTIONS, _CONSTRAINTS,
            dict_name='scheduler_options')
        resume = kwargs['resume']

        self.ema_baseline_decay = kwargs['ema_baseline_decay']
        self.sync = kwargs['sync']
        # create RL searcher if not passed
        searcher = kwargs.get('searcher')
        if not isinstance(searcher, RLSearcher):
            if searcher is not None:
                logger.warning("Argument 'searcher' must be of type RLSearcher. Ignoring 'searcher' and creating searcher here.")
            kwargs['searcher'] = RLSearcher(
                train_fn.kwspaces, reward_attribute=kwargs['reward_attr'])
        # Pass resume=False here. Resume needs members of this object to be
        # created
        kwargs['resume'] = False
        super().__init__(
            train_fn=train_fn, **filter_by_key(kwargs, _ARGUMENT_KEYS))
        # reserve controller computation resource on master node
        master_node = self.remote_manager.get_master_node()
        controller_resource = kwargs['controller_resource']
        self.controller_resource = DistributedResource(**controller_resource)
        assert self.resource_manager.reserve_resource(
            master_node, self.controller_resource),\
            "Not Enough Resource on Master Node for Training Controller"
        if controller_resource['num_gpus'] > 0:
            self.controller_ctx = [
                mx.gpu(i) for i in self.controller_resource.gpu_ids]
        else:
            self.controller_ctx = [mx.cpu()]
        # controller setup
        self.controller = self.searcher.controller
        self.controller.collect_params().reset_ctx(self.controller_ctx)
        controller_batch_size = kwargs['controller_batch_size']
        learning_rate = kwargs['controller_lr'] * controller_batch_size
        self.controller_optimizer = mx.gluon.Trainer(
            self.controller.collect_params(), 'adam',
            optimizer_params={'learning_rate': learning_rate})
        self.controller_batch_size = controller_batch_size
        self.baseline = None
        self.lock = mp.Lock()
        # async buffers
        if not self.sync:
            self.mp_count = mp.Value('i', 0)
            self.mp_seed = mp.Value('i', 0)
            self.mp_fail = mp.Value('i', 0)
        if resume:
            checkpoint = kwargs.get('checkpoint')
            if os.path.isfile(checkpoint):
                self.load_state_dict(load(checkpoint))
            else:
                msg = 'checkpoint path {} is not available for resume.'.format(checkpoint)
                logger.exception(msg)

    def run(self, **kwargs):
        """Run multiple number of trials
        """
        self.num_trials = kwargs.get('num_trials', self.num_trials)
        logger.info('Starting Experiments')
        logger.info('Num of Finished Tasks is {}'.format(self.num_finished_tasks))
        logger.info('Num of Pending Tasks is {}'.format(self.num_trials - self.num_finished_tasks))
        if self.sync:
            self._run_sync()
        else:
            self._run_async()

    def _run_sync(self):
        try_import_mxnet()
        import mxnet as mx

        decay = self.ema_baseline_decay
        for i in tqdm(range(self.num_trials // self.controller_batch_size + 1)):
            with mx.autograd.record():
                # sample controller_batch_size number of configurations
                batch_size = self.num_trials % self.num_trials \
                    if i == self.num_trials // self.controller_batch_size \
                    else self.controller_batch_size
                if batch_size == 0: continue
                configs, log_probs, entropies = self.controller.sample(
                    batch_size, with_details=True)
                # schedule the training tasks and gather the reward
                rewards = self.sync_schedule_tasks(configs)
                # substract baseline
                if self.baseline is None:
                    self.baseline = rewards[0]
                avg_rewards = mx.nd.array([reward - self.baseline for reward in rewards],
                                          ctx=self.controller.context)
                # EMA baseline
                for reward in rewards:
                    self.baseline = decay * self.baseline + (1 - decay) * reward
                # negative policy gradient
                log_probs = log_probs.sum(axis=1)
                loss = - log_probs * avg_rewards#.reshape(-1, 1)
                loss = loss.sum()  # or loss.mean()

            # update
            loss.backward()
            self.controller_optimizer.step(batch_size)
            logger.debug('controller loss: {}'.format(loss.asscalar()))

    def _run_async(self):
        try_import_mxnet()
        import mxnet as mx

        def _async_run_trial():
            self.mp_count.value += 1
            self.mp_seed.value += 1
            seed = self.mp_seed.value
            mx.random.seed(seed)
            with mx.autograd.record():
                # sample one configuration
                with self.lock:
                    config, log_prob, entropy = self.controller.sample(with_details=True)
                config = config[0]
                task = Task(self.train_fn, {'args': self.args, 'config': config},
                            DistributedResource(**self.resource))
                # start training task
                reporter = DistStatusReporter(remote=task.resources.node)
                task.args['reporter'] = reporter
                task_thread = self.add_job(task)

                # run reporter
                last_result = None
                config = task.args['config']
                while task_thread.is_alive():
                    reported_result = reporter.fetch()
                    if reported_result.get('done', False):
                        reporter.move_on()
                        task_thread.join()
                        break
                    self._add_training_result(task.task_id, reported_result, task.args['config'])
                    reporter.move_on()
                    last_result = reported_result
                self.searcher.update(config, **last_result)
                reward = last_result[self._reward_attr]
                with self.lock:
                    if self.baseline is None:
                        self.baseline = reward
                avg_reward = mx.nd.array([reward - self.baseline], ctx=self.controller.context)
                # negative policy gradient
                with self.lock:
                    loss = -log_prob * avg_reward.reshape(-1, 1)
                    loss = loss.sum()

            # update
            print('loss', loss)
            with self.lock:
                try:
                    loss.backward()
                    self.controller_optimizer.step(1)
                except Exception:
                    self.mp_fail.value += 1
                    logger.warning('Exception during backward {}.'.format(self.mp_fail.value))

            self.mp_count.value -= 1
            # ema
            with self.lock:
                decay = self.ema_baseline_decay
                self.baseline = decay * self.baseline + (1 - decay) * reward

        reporter_threads = []
        for i in range(self.num_trials):
            while self.mp_count.value >= self.controller_batch_size:
                time.sleep(0.2)
            #_async_run_trial()
            reporter_thread = threading.Thread(target=_async_run_trial)
            reporter_thread.start()
            reporter_threads.append(reporter_thread)

        for p in reporter_threads:
            p.join()

    def sync_schedule_tasks(self, configs):
        rewards = []
        results = {}
        def _run_reporter(task, task_job, reporter):
            last_result = None
            config = task.args['config']
            while not task_job.done():
                reported_result = reporter.fetch()
                if 'traceback' in reported_result:
                    logger.exception(reported_result['traceback'])
                    reporter.move_on()
                    break

                if reported_result.get('done', False):
                    reporter.move_on()
                    break
                self._add_training_result(task.task_id, reported_result, task.args['config'])
                reporter.move_on()
                last_result = reported_result
            if last_result is not None:
                self.searcher.update(config, **last_result)
                with self.lock:
                    results[pickle.dumps(config)] = \
                        last_result[self._reward_attr]

        # launch the tasks
        tasks = []
        task_jobs = []
        reporter_threads = []
        for config in configs:
            logger.debug('scheduling config: {}'.format(config))
            # create task
            task = Task(self.train_fn, {'args': self.args, 'config': config},
                        DistributedResource(**self.resource))
            reporter = DistStatusReporter()
            task.args['reporter'] = reporter
            task_job = self.add_job(task)
            # run reporter
            reporter_thread = threading.Thread(target=_run_reporter, args=(task, task_job, reporter))
            reporter_thread.start()
            tasks.append(task)
            task_jobs.append(task_job)
            reporter_threads.append(reporter_thread)

        for p1, p2 in zip(task_jobs, reporter_threads):
            p1.result()
            p2.join()
        with self.LOCK:
            for task in tasks:
                self.finished_tasks.append({'TASK_ID': task.task_id,
                                           'Config': task.args['config']})
        if self._checkpoint is not None:
            logger.debug('Saving Checkerpoint')
            self.save()

        for config in configs:
            rewards.append(results[pickle.dumps(config)])

        return rewards

    def add_job(self, task, **kwargs):
        """Adding a training task to the scheduler.

        Args:
            task (:class:`autogluon.scheduler.Task`): a new training task
        """
        cls = RLScheduler
        cls.resource_manager._request(task.resources)
        # main process
        job = cls._start_distributed_job(task, cls.resource_manager)
        return job

    def join_tasks(self):
        pass

    def state_dict(self, destination=None):
        """Returns a dictionary containing a whole state of the Scheduler

        Examples
        --------
        >>> ag.save(scheduler.state_dict(), 'checkpoint.ag')
        """
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        logger.debug('\nState_Dict self.finished_tasks: {}'.format(self.finished_tasks))
        destination['finished_tasks'] = pickle.dumps(self.finished_tasks)
        destination['baseline'] = pickle.dumps(self.baseline)
        destination['TASK_ID'] = Task.TASK_ID.value
        destination['searcher'] = self.searcher.state_dict()
        destination['training_history'] = json.dumps(self.training_history)
        if self.visualizer == 'mxboard' or self.visualizer == 'tensorboard':
            destination['visualizer'] = json.dumps(self.mxboard._scalar_dict)
        return destination

    def load_state_dict(self, state_dict):
        """Load from the saved state dict.

        Examples
        --------
        >>> scheduler.load_state_dict(ag.load('checkpoint.ag'))
        """
        self.finished_tasks = pickle.loads(state_dict['finished_tasks'])
        #self.baseline = pickle.loads(state_dict['baseline'])
        Task.set_id(state_dict['TASK_ID'])
        self.searcher.load_state_dict(state_dict['searcher'])
        self.training_history = json.loads(state_dict['training_history'])
        if self.visualizer == 'mxboard' or self.visualizer == 'tensorboard':
            self.mxboard._scalar_dict = json.loads(state_dict['visualizer'])
        logger.debug('Loading Searcher State {}'.format(self.searcher))
