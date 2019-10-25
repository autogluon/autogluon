import os
import time
import copy
import pickle
import json
import logging
import threading
from tqdm import trange
import multiprocessing as mp
from collections import OrderedDict

from .resource import DistributedResource
from ..utils import save, load, mkdir, try_import_mxboard
from ..core import Task
from ..core.decorator import _autogluon_method
from .scheduler import TaskScheduler
from ..searcher import *
from .reporter import DistStatusReporter
from ..utils import DeprecationHelper

__all__ = ['FIFOScheduler', 'DistributedFIFOScheduler']

logger = logging.getLogger(__name__)

searchers = {
    'random': RandomSearcher,
    'skopt': SKoptSearcher,  # May have other BO solutions in the future...
    'grid': GridSearcher,
}

class FIFOScheduler(TaskScheduler):
    """Simple scheduler that just runs trials in submission order.

    Args:
        train_fn (callable): A task launch function for training. Note: please add the `@autogluon_method` decorater to the original function.
        args (object): Default arguments for launching train_fn.
        resource (dict): Computation resources. For example, `{'num_cpus':2, 'num_gpus':1}`
        searcher (str or object): Autogluon searcher. For example, autogluon.searcher.self.argsRandomSampling
        reward_attr (str): The training result objective value attribute. As with `time_attr`, this may refer to any objective value. Stopping procedures will use this attribute.

    Example:
        >>> @autogluon_method
        >>> def train_fn(args, reporter):
        >>>     for e in range(10):
        >>>         # forward, backward, optimizer step and evaluation metric
        >>>         # generate fake top1_accuracy
        >>>         top1_accuracy = 1 - np.power(1.8, -np.random.uniform(e, 2*e))
        >>>         reporter(epoch=e, accuracy=top1_accuracy)
        >>> import ConfigSpace as CS
        >>> import ConfigSpace.hyperparameters as CSH
        >>> cs = CS.ConfigurationSpace()
        >>> lr = CSH.UniformFloatHyperparameter('lr', lower=1e-4, upper=1e-1, log=True)
        >>> cs.add_hyperparameter(lr)
        >>> searcher = RandomSampling(cs)
        >>> myscheduler = FIFOScheduler(train_fn, args,
        >>>                             resource={'num_cpus': 2, 'num_gpus': 0},
        >>>                             searcher=searcher, num_trials=20,
        >>>                             reward_attr='accuracy',
        >>>                             time_attr='epoch',
        >>>                             grace_period=1)
        >>> # run tasks
        >>> myscheduler.run()
    """
    def __init__(self, train_fn, args=None, resource=None,
                 searcher='random', search_options=None,
                 checkpoint='./exp/checkerpoint.ag',
                 resume=False, num_trials=None,
                 time_out=None, max_reward=1.0, time_attr='epoch',
                 reward_attr='accuracy',
                 visualizer='none', dist_ip_addrs=None):
        super(FIFOScheduler,self).__init__(dist_ip_addrs)
        if resource is None:
            resource = {'num_cpus': 1, 'num_gpus': 0}
        if search_options is None:
            search_options = dict()
        assert isinstance(train_fn, _autogluon_method)
        self.train_fn = train_fn
        self.args = args if args else train_fn.args
        self.resource = resource
        if isinstance(searcher, str):
            self.searcher = searchers[searcher](train_fn.cs, **search_options)
        else:
            assert isinstance(searcher, BaseSearcher)
            self.searcher = searcher
        # meta data
        self.metadata = train_fn.kwspaces
        keys = copy.deepcopy(list(self.metadata.keys()))
        for k in keys:
            if '.' in k:
                v = self.metadata.pop(k)
                new_k = k.split('.')[-1]
                self.metadata[new_k] = v
        self.metadata['search_strategy'] = searcher
        self.metadata['stop_criterion'] = {'time_limits': time_out, 'max_reward': max_reward}
        self.metadata['resources_per_trial'] = resource

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
                verbose=False)
        self.log_lock = mp.Lock()
        self.training_history = OrderedDict()
        self.config_history = OrderedDict()

        if resume:
            if os.path.isfile(checkpoint):
                self.load_state_dict(load(checkpoint))
            else:
                msg = 'checkpoint path {} is not available for resume.'.format(checkpoint)
                logger.exception(msg)
                raise FileExistsError(msg)

    def run(self, **kwargs):
        """Run multiple number of trials
        """
        start_time = time.time()
        self.num_trials = kwargs.get('num_trials', self.num_trials)
        self.time_out = kwargs.get('time_out', self.time_out)
        logger.info('Starting Experiments')
        logger.info('Num of Finished Tasks is {}'.format(self.num_finished_tasks))
        logger.info('Num of Pending Tasks is {}'.format(self.num_trials - self.num_finished_tasks))
        tbar = trange(self.num_finished_tasks, self.num_trials)
        for _ in tbar:
            if self.time_out and time.time() - start_time >= self.time_out \
                    or self.max_reward and self.get_best_reward() >= self.max_reward:
                break
            tbar.set_description('Current best reward: {} '.format(self.get_best_reward()))
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
            mkdir(os.path.dirname(checkpoint))
            save(self.state_dict(), checkpoint)

    def schedule_next(self):
        """Schedule next searcher suggested task
        """
        # Allow for the promotion of a previously chosen config. Also,
        # extra_kwargs contains extra info passed to both add_task and to
        # get_config (if no config is promoted)
        config, extra_kwargs = self._promote_config()
        if config is None:
            # No config to promote: Query next config to evaluate from searcher
            config = self.searcher.get_config(**extra_kwargs)
            extra_kwargs['new_config'] = True
        else:
            # This is not a new config, but a paused one which is now promoted
            extra_kwargs['new_config'] = False
        task = Task(self.train_fn, {'args': self.args, 'config': config},
                    DistributedResource(**self.resource))
        self.add_task(task, **extra_kwargs)

    def _dict_from_task(self, task):
        if isinstance(task, Task):
            return {'TASK_ID': task.task_id, 'Config': task.args['config']}
        else:
            assert isinstance(task, dict)
            return {'TASK_ID': task['TASK_ID'], 'Config': task['Config']}

    def add_task(self, task, **kwargs):
        """Adding a training task to the scheduler.

        Args:
            task (:class:`autogluon.scheduler.Task`): a new trianing task
        """
        cls = FIFOScheduler
        cls.RESOURCE_MANAGER._request(task.resources)
        # reporter
        reporter = DistStatusReporter()
        task.args['reporter'] = reporter
        # Register pending evaluation
        self.searcher.register_pending(task.args['config'])
        # main process
        job = cls._start_distributed_task(task, cls.RESOURCE_MANAGER, self.env_sem)
        # reporter thread
        rp = threading.Thread(target=self._run_reporter, args=(task, job, reporter,
                              self.searcher), daemon=False)
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

    def _run_reporter(self, task, task_job, reporter, searcher,
                      checkpoint_semaphore=None):
        last_result = None
        while not task_job.done():
            reported_result = reporter.fetch()
            if reported_result.get('done', False):
                reporter.move_on()
                if checkpoint_semaphore is not None:
                    checkpoint_semaphore.release()
                break
            self.add_training_result(
                task.task_id, reported_result, config=task.args['config'])
            reporter.move_on()
            last_result = reported_result
        if last_result is not None:
            last_result['done'] = True
            searcher.update(
                config=task.args['config'],
                reward=last_result[self._reward_attr], **last_result)

    def _promote_config(self):
        """
        Provides a hook in schedule_next, which allows to promote a config
        which has been selected and partially evaluated previously.

        :return: config, extra_args
        """
        config = None
        extra_args = dict()
        return config, extra_args

    def get_best_state(self):
        raise NotImplemented

    def get_best_config(self):
        # Enable interactive monitoring
        return self.searcher.get_best_config()

    def get_best_reward(self):
        # Enable interactive monitoring
        return self.searcher.get_best_reward()

    def add_training_result(self, task_id, reported_result, config=None):
        if self.visualizer == 'mxboard' or self.visualizer == 'tensorboard':
            if 'loss' in reported_result:
                self.mxboard.add_scalar(tag='loss',
                                        value=('task {task_id} valid_loss'.format(task_id=task_id),
                                               reported_result['loss']),
                                        global_step=reported_result['epoch'])
            self.mxboard.add_scalar(tag=self._reward_attr,
                                    value=('task {task_id} {reward_attr}'.format(
                                           task_id=task_id, reward_attr=self._reward_attr),
                                           reported_result[self._reward_attr]),
                                    global_step=reported_result['epoch'])
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
        if filename is None and not plot:
            logger.warning('Please either provide filename or allow plot in get_training_curves')
        import matplotlib.pyplot as plt
        plt.ylabel(self._reward_attr)
        plt.xlabel(self._time_attr)
        for task_id, task_res in self.training_history.items():
            rewards = [x[self._reward_attr] for x in task_res]
            x = list(range(len(task_res)))
            plt.plot(x, rewards, label='task {}'.format(task_id))
        if use_legend:
            plt.legend(loc='best')
        if filename is not None:
            logger.info('Saving Training Curve in {}'.format(filename))
            plt.savefig(filename)
        if plot: plt.show()

    def state_dict(self, destination=None):
        destination = super(FIFOScheduler, self).state_dict(destination)
        destination['searcher'] = pickle.dumps(self.searcher)
        destination['training_history'] = json.dumps(self.training_history)
        if self.visualizer == 'mxboard' or self.visualizer == 'tensorboard':
            destination['visualizer'] = json.dumps(self.mxboard._scalar_dict)
        return destination

    def load_state_dict(self, state_dict):
        super(FIFOScheduler, self).load_state_dict(state_dict)
        self.searcher = pickle.loads(state_dict['searcher'])
        self.training_history = json.loads(state_dict['training_history'])
        if self.visualizer == 'mxboard' or self.visualizer == 'tensorboard':
            self.mxboard._scalar_dict = json.loads(state_dict['visualizer'])
        logger.debug('Loading Searcher State {}'.format(self.searcher))


DistributedFIFOScheduler = DeprecationHelper(FIFOScheduler, 'DistributedFIFOScheduler')

