import json
import logging
import multiprocessing as mp
import os
import pickle
import threading
import time
from collections import OrderedDict

from tqdm.auto import tqdm

from .reporter import DistStatusReporter, FakeReporter
from .resource import DistributedResource
from .scheduler import TaskScheduler
from ..core import Task
from ..core.decorator import _autogluon_method
from ..searcher import BaseSearcher
from ..searcher.searcher_factory import searcher_factory
from ..utils import save, load, mkdir, try_import_mxboard

__all__ = ['FIFOScheduler']

logger = logging.getLogger(__name__)


class FIFOScheduler(TaskScheduler):
    r"""Simple scheduler that just runs trials in submission order.

    Parameters
    ----------
    train_fn : callable
        A task launch function for training. Note: please add the `@autogluon_method` decorater to the original function.
    args : object (optional)
        Default arguments for launching train_fn.
    resource : dict
        Computation resources. For example, `{'num_cpus':2, 'num_gpus':1}`
    searcher : str or object
        Autogluon searcher. For example, autogluon.searcher.self.argsRandomSampling
    time_attr : str
            A training result attr to use for comparing time.
            Note that you can pass in something non-temporal such as
            `training_epoch` as a measure of progress, the only requirement
            is that the attribute should increase monotonically.
    reward_attr : str
        The training result objective value attribute. As with `time_attr`, this may refer to any objective value.
        Stopping procedures will use this attribute.
    dist_ip_addrs : list of str
        IP addresses of remote machines.

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
    ...         reporter(epoch=e, accuracy=dummy_accuracy, lr=args.lr, wd=args.wd)
    >>> scheduler = ag.scheduler.FIFOScheduler(train_fn,
    ...                                        resource={'num_cpus': 2, 'num_gpus': 0},
    ...                                        num_trials=20,
    ...                                        reward_attr='accuracy',
    ...                                        time_attr='epoch')
    >>> scheduler.run()
    >>> scheduler.join_jobs()
    >>> scheduler.get_training_curves(plot=True)
    """

    def __init__(self, train_fn, args=None, resource=None,
                 searcher=None, search_options=None,
                 checkpoint='./exp/checkpoint.ag',
                 resume=False, num_trials=None,
                 time_out=None, max_reward=1.0, time_attr='epoch',
                 reward_attr='accuracy',
                 visualizer='none', dist_ip_addrs=None):
        super().__init__(dist_ip_addrs)

        if resource is None:
            resource = {'num_cpus': 1, 'num_gpus': 0}
        self.resource = resource

        if searcher is None:
            searcher = 'random'  # Default: Random searcher

        if isinstance(searcher, str):
            kwargs = search_options.copy() if search_options else dict()
            kwargs['configspace'] = train_fn.cs
            self.searcher: BaseSearcher = searcher_factory(searcher, **kwargs)
        else:
            assert isinstance(searcher, BaseSearcher)
            self.searcher: BaseSearcher = searcher

        assert isinstance(train_fn, _autogluon_method)
        self.train_fn = train_fn
        self.args = args if args else train_fn.args

        # meta data
        self.metadata = {
            'search_space': train_fn.kwspaces,
            'search_strategy': searcher,
            'stop_criterion': {'time_limits': time_out, 'max_reward': max_reward},
            'resources_per_trial': resource
        }

        self.num_trials = num_trials
        self.time_out = time_out
        self.max_reward = max_reward
        self._checkpoint = checkpoint
        self._time_attr = time_attr
        self._reward_attr = reward_attr
        self.visualizer = visualizer.lower()
        if self.visualizer == 'tensorboard' or self.visualizer == 'mxboard':
            try_import_mxboard()
            from mxboard import SummaryWriter
            self.mxboard = SummaryWriter(
                logdir=os.path.join(os.path.splitext(checkpoint)[0], 'logs'),
                flush_secs=3,
                verbose=False
            )
        self.log_lock = mp.Lock()
        self.training_history = OrderedDict()
        self.config_history = OrderedDict()

        if resume:
            if os.path.isfile(checkpoint):
                self.load_state_dict(load(checkpoint))
            else:
                msg = f'checkpoint path {checkpoint} is not available for resume.'
                logger.exception(msg)
                raise FileExistsError(msg)

    def run(self, **kwargs):
        """Run multiple number of trials
        """
        start_time = time.time()
        num_trials = kwargs.get('num_trials', self.num_trials)
        time_out = kwargs.get('time_out', self.time_out)
        logger.info('Starting Experiments')
        logger.info(f'Num of Finished Tasks is {self.num_finished_tasks}')
        logger.info(f'Num of Pending Tasks is {num_trials - self.num_finished_tasks}')
        tbar = tqdm(range(self.num_finished_tasks, num_trials))
        for _ in tbar:
            if time_out and time.time() - start_time >= time_out \
                    or self.max_reward and self.get_best_reward() >= self.max_reward:
                break
            self.schedule_next()

    def save(self, checkpoint=None):
        """Save Checkpoint
        """
        if checkpoint is None:
            if self._checkpoint is None:
                logger.warning("Checkpointing is disabled")
            else:
                checkpoint = self._checkpoint
        if checkpoint is not None:
            with self.LOCK:
                mkdir(os.path.dirname(checkpoint))
                save(self.state_dict(), checkpoint)

    def schedule_next(self):
        """Schedule next searcher suggested task
        """
        # Allow for the promotion of a previously chosen config. Also,
        # extra_kwargs contains extra info passed to both add_job and to
        # get_config (if no config is promoted)

        # request resource first
        resources = DistributedResource(**self.resource)
        FIFOScheduler.RESOURCE_MANAGER._request(resources)

        config, extra_kwargs = self._promote_config()
        if config is None:
            # No config to promote: Query next config to evaluate from searcher
            config = self.searcher.get_config(**extra_kwargs)
            extra_kwargs['new_config'] = True
        else:
            # This is not a new config, but a paused one which is now promoted
            extra_kwargs['new_config'] = False
        task = Task(
            self.train_fn,
            {'args': self.args, 'config': config},
            resources
        )
        self.add_job(task, **extra_kwargs)

    def run_with_config(self, config):
        """Run with config for final fit.
        It launches a single training trial under any fixed values of the hyperparameters.
        For example, after HPO has identified the best hyperparameter values based on a hold-out dataset,
        one can use this function to retrain a model with the same hyperparameters on all the available labeled data
        (including the hold out set). It can also returns other objects or states.
        """
        task = Task(
            self.train_fn,
            {'args': self.args, 'config': config},
            DistributedResource(**self.resource)
        )
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
        if not task.resources.is_ready:
            cls.RESOURCE_MANAGER._request(task.resources)
        # reporter
        reporter = DistStatusReporter(remote=task.resources.node)
        task.args['reporter'] = reporter
        # Register pending evaluation
        self.searcher.register_pending(task.args['config'])
        # main process
        job = cls._start_distributed_job(task, cls.RESOURCE_MANAGER)
        # reporter thread
        rp = threading.Thread(
            target=self._run_reporter,
            args=(task, job, reporter, self.searcher),
            daemon=False
        )
        rp.start()
        task_dict = self._dict_from_task(task)
        task_dict.update({'Task': task, 'Job': job, 'ReporterThread': rp})
        # checkpoint thread
        if self._checkpoint is not None:
            def _save_checkpoint_callback(fut):
                self._cleaning_tasks()
                self.save()

            job.add_done_callback(_save_checkpoint_callback)

        with self.LOCK:
            self.scheduled_tasks.append(task_dict)

    def _clean_task_internal(self, task_dict):
        task_dict['ReporterThread'].join()

    def _run_reporter(self, task, task_job, reporter, searcher):
        last_result = None
        while not task_job.done():
            reported_result = reporter.fetch()
            if 'traceback' in reported_result:
                logger.exception(reported_result['traceback'])
                reporter.move_on()
                break

            if reported_result.get('done', False):
                reporter.move_on()
                break
            self._add_training_result(
                task.task_id, reported_result, config=task.args['config'])
            reporter.move_on()
            last_result = reported_result
        if last_result is not None:
            last_result['done'] = True
            searcher.update(
                config=task.args['config'],
                reward=last_result[self._reward_attr],
                **last_result
            )

    def _promote_config(self):
        """
        Provides a hook in schedule_next, which allows to promote a config
        which has been selected and partially evaluated previously.

        :return: config, extra_args
        """
        config = None
        extra_args = dict()
        return config, extra_args

    def get_best_config(self):
        """Get the best configuration from the finished jobs.
        """
        return self.searcher.get_best_config()

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
                    global_step=reported_result[self._reward_attr]
                )
            self.mxboard.add_scalar(
                tag=self._reward_attr,
                value=(
                    f'task {task_id} {self._reward_attr}',
                    reported_result[self._reward_attr]
                ),
                global_step=reported_result[self._reward_attr]
            )
        with self.log_lock:
            # Note: We store all of reported_result in training_history[task_id],
            # not just the reward value.
            if task_id in self.training_history:
                self.training_history[task_id].append(reported_result)
            else:
                self.training_history[task_id] = [reported_result]
                if config:
                    self.config_history[task_id] = config

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
        plt.ylabel(self._reward_attr)
        plt.xlabel(self._time_attr)
        plt.title("Performance vs Training-Time in each HPO Trial")
        with self.log_lock:
            for task_id, task_res in self.training_history.items():
                rewards = [x[self._reward_attr] for x in task_res]
                x = list(range(len(task_res)))
                plt.plot(x, rewards, label=f'task {task_id}')
        if use_legend:
            plt.legend(loc='best')
        if filename:
            logger.info(f'Saving Training Curve in {filename}')
            plt.savefig(filename)
        if plot:
            plt.show()

    def state_dict(self, destination=None):
        """Returns a dictionary containing a whole state of the Scheduler

        Examples
        --------
        >>> ag.save(scheduler.state_dict(), 'checkpoint.ag')
        """
        destination = super().state_dict(destination)
        destination['searcher'] = pickle.dumps(self.searcher)
        with self.log_lock:
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
        super().load_state_dict(state_dict)
        self.searcher = pickle.loads(state_dict['searcher'])
        with self.log_lock:
            self.training_history = json.loads(state_dict['training_history'])
        if self.visualizer == 'mxboard' or self.visualizer == 'tensorboard':
            self.mxboard._scalar_dict = json.loads(state_dict['visualizer'])
        logger.debug(f'Loading Searcher State {self.searcher}')
