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
from ..core import Task, autogluon_method
from .scheduler import TaskScheduler
from ..searcher import *
from .reporter import DistStatusReporter
from ..utils import DeprecationHelper

__all__ = ['FIFOScheduler', 'DistributedFIFOScheduler']

logger = logging.getLogger(__name__)

searchers = {
    'hyperband': RandomSearcher,
    'random': RandomSearcher,
    'bayesopt': SKoptSearcher,
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
    def __init__(self, train_fn, args=None, resource={'num_cpus': 1, 'num_gpus': 0},
                 searcher='random', search_options={}, checkpoint='./exp/checkerpoint.ag',
                 resume=False, num_trials=None,
                 time_out=None, max_reward=1.0, time_attr='epoch', reward_attr='accuracy',
                 visualizer='none', dist_ip_addrs=[], **kwargs):
        super(FIFOScheduler,self).__init__(dist_ip_addrs)
        self.train_fn = train_fn
        assert isinstance(train_fn, autogluon_method)
        self.args = args if args else train_fn.args
        self.resource = resource
        self.searcher = searchers[searcher](train_fn.cs, **search_options) if isinstance(searcher, str) else searcher
        # meta data
        self.metadata = train_fn.get_kwspaces()
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

    def run(self, num_trials=None, time_out=None):
        """Run multiple number of trials
        """
        start_time = time.time()
        self.num_trials = num_trials if num_trials else self.num_trials
        self.time_out = time_out if time_out else self.time_out
        logger.info('Starting Experiments')
        logger.info('Num of Finished Tasks is {}'.format(self.num_finished_tasks))
        logger.info('Num of Pending Tasks is {}'.format(self.num_trials - self.num_finished_tasks))
        tbar = trange(self.num_finished_tasks, self.num_trials)
        for _ in tbar:
            if time_out and time.time() - start_time >= time_out \
                    or self.max_reward and self.get_best_reward() >= self.max_reward:
                break
            tbar.set_description('Current best reward: {} and best config: {}'
                                 .format(self.get_best_reward(), json.dumps(self.get_best_config())))
            self.schedule_next()

    def save(self, checkpoint=None):
        """Save Checkpoint
        """
        if checkpoint is None and self._checkpoint is None:
            msg = 'Please set checkpoint path.'
            logger.exception(msg)
            raise RuntimeError(msg)
        checkname = checkpoint if checkpoint else self._checkpoint
        mkdir(os.path.dirname(checkname))
        save(self.state_dict(), checkname)

    def schedule_next(self):
        """Schedule next searcher suggested task
        """
        task = Task(self.train_fn, {'args': self.args, 'config': self.searcher.get_config()},
                    DistributedResource(**self.resource))
        self.add_task(task)

    def add_task(self, task):
        """Adding a training task to the scheduler.

        Args:
            task (:class:`autogluon.scheduler.Task`): a new trianing task
        """
        cls = FIFOScheduler
        cls.RESOURCE_MANAGER._request(task.resources)
        # reporter
        reporter = DistStatusReporter()
        task.args['reporter'] = reporter
        # main process
        tp = threading.Thread(target=cls._start_distributed_task, args=(
                              task, cls.RESOURCE_MANAGER, self.env_sem))
        # reporter thread
        checkpoint_semaphore = mp.Semaphore(0) if self._checkpoint else None
        rp = threading.Thread(target=self._run_reporter, args=(task, tp, reporter,
                              self.searcher, checkpoint_semaphore), daemon=False)
        tp.start()
        rp.start()
        task_dict = {'TASK_ID': task.task_id, 'Config': task.args['config'], 'Task': task,
                     'Process': tp, 'ReporterThread': rp}
        # checkpoint thread
        if self._checkpoint is not None:
            sp = threading.Thread(target=self._run_checkpoint, args=(checkpoint_semaphore,),
                                  daemon=False)
            sp.start()
            task_dict['CheckpointThead'] = sp

        with self.LOCK:
            self.scheduled_tasks.append(task_dict)

    def join_tasks(self):
        self._cleaning_tasks()
        for i, task_dict in enumerate(self.scheduled_tasks):
            task_dict['Process'].join()
            task_dict['ReporterThread'].join()

    def _cleaning_tasks(self):
        with self.LOCK:
            for i, task_dict in enumerate(self.scheduled_tasks):
                if not task_dict['Process'].is_alive():
                    task_dict = self.scheduled_tasks.pop(i)
                    task_dict['ReporterThread'].join()
                    self.finished_tasks.append({'TASK_ID': task_dict['TASK_ID'],
                                               'Config': task_dict['Config']})

    def _run_checkpoint(self, checkpoint_semaphore):
        self._cleaning_tasks()
        checkpoint_semaphore.acquire()
        logger.debug('Saving Checkerpoint')
        self.save()

    def _run_reporter(self, task, task_process, reporter, searcher,
                      checkpoint_semaphore):
        last_result = None
        while task_process.is_alive():
            reported_result = reporter.fetch()
            if 'done' in reported_result and reported_result['done'] is True:
                reporter.move_on()
                task_process.join()
                if checkpoint_semaphore is not None:
                    checkpoint_semaphore.release()
                break
            self.add_training_result(task.task_id, reported_result, task.args['config'])
            reporter.move_on()
            last_result = reported_result
        if last_result is not None:
            searcher.update(task.args['config'], last_result[self._reward_attr])

    def get_best_state(self):
        raise NotImplemented

    def get_best_config(self):
        # Enable interactive monitoring
        # self.join_tasks()
        return self.searcher.get_best_config()

    def get_best_reward(self):
        # Enable interactive monitoring
        # self.join_tasks()
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
        reward = reported_result[self._reward_attr]
        with self.log_lock:
            if task_id in self.training_history:
                self.training_history[task_id].append(reward)
            else:
                self.training_history[task_id] = [reward]
                if config:
                    self.config_history[task_id] = config

    def get_training_curves(self, filename=None, plot=False, use_legend=True):
        if filename is None and not plot:
            logger.warning('Please either provide filename or allow plot in get_training_curves')
        import matplotlib.pyplot as plt
        plt.ylabel(self._reward_attr)
        plt.xlabel(self._time_attr)
        for task_id, task_res in self.training_history.items():
            x = list(range(len(task_res)))
            plt.plot(x, task_res, label='task {}'.format(task_id))
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

