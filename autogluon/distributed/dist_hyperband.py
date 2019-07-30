import pickle
import logging
import threading
import numpy as np
import multiprocessing as mp

from ..basic import Task
from ..scheduler.hyperband import Hyperband_Manager
from .dist_fifo import DistributedFIFOScheduler
from .dist_reporter import DistStatusReporter, DistSemaphore

__all__ = ['DistributedHyperbandScheduler']

logger = logging.getLogger(__name__)

# Async version of Hyperband used in computation heavy tasks such as deep learning
class DistributedHyperbandScheduler(DistributedFIFOScheduler):
    """Implements the Async Hyperband
    This should provide similar theoretical performance as HyperBand but
    avoid straggler issues that HyperBand faces. One implementation detail
    is when using multiple brackets, task allocation to bracket is done
    randomly with over a softmax probability.
    See https://arxiv.org/abs/1810.05934

    Args:
        train_fn (callable): A task launch function for training.
            Note: please add the `@autogluon_method` decorater to the original function.
        args (object): Default arguments for launching train_fn.
        resource (dict): Computation resources.
            For example, `{'num_cpus':2, 'num_gpus':1}`
        searcher (object): Autogluon searcher.
            For example, autogluon.searcher.RandomSampling
        time_attr (str): A training result attr to use for comparing time.
            Note that you can pass in something non-temporal such as
            `training_epoch` as a measure of progress, the only requirement
            is that the attribute should increase monotonically.
        reward_attr (str): The training result objective value attribute. As
            with `time_attr`, this may refer to any objective value. Stopping
            procedures will use this attribute.
        max_t (float): max time units per task. Trials will be stopped after
            max_t time units (determined by time_attr) have passed.
        grace_period (float): Only stop tasks at least this old in time.
            The units are the same as the attribute named by `time_attr`.
        reduction_factor (float): Used to set halving rate and amount. This
            is simply a unit-less scalar.
        brackets (int): Number of brackets. Each bracket has a different
            halving rate, specified by the reduction factor.

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
        >>> myscheduler = DistributedHyperbandScheduler(train_fn, args,
        >>>                                   resource={'num_cpus': 2, 'num_gpus': 0}, 
        >>>                                   searcher=searcher, num_trials=20,
        >>>                                   reward_attr='accuracy',
        >>>                                   time_attr='epoch',
        >>>                                   grace_period=1)
    """
    def __init__(self, train_fn, args, resource, searcher, checkpoint='./exp/checkerpoint.ag',
                 resume=False, num_trials=None, time_attr="epoch", reward_attr="accuracy",
                 max_t=100, grace_period=10, reduction_factor=4, brackets=1,
                 visualizer='none', dist_ip_addrs=[]):
        super(DistributedHyperbandScheduler, self).__init__(
            train_fn, args, resource, searcher, checkpoint, resume, num_trials,
            time_attr, reward_attr, visualizer, dist_ip_addrs)
        self.terminator = Hyperband_Manager(time_attr, reward_attr, max_t, grace_period,
                                            reduction_factor, brackets)

    def add_task(self, task):
        """Adding a training task to the scheduler.

        Args:
            task (:class:`autogluon.scheduler.Task`): a new trianing task
        """
        cls = DistributedHyperbandScheduler
        cls.RESOURCE_MANAGER._request(task.resources)
        # reporter and terminator
        reporter = DistStatusReporter()
        terminator_semaphore = DistSemaphore(0)
        task.args['reporter'] = reporter
        task.args['terminator_semaphore'] = terminator_semaphore
        self.terminator.on_task_add(task)
        # main process
        tp = threading.Thread(target=cls._start_distributed_task, args=(
                              task, cls.RESOURCE_MANAGER, self.env_sem))
        # reporter thread
        checkpoint_semaphore = mp.Semaphore(0) if self._checkpoint else None
        rp = threading.Thread(target=self._run_reporter,
                              args=(task, tp, reporter, self.searcher, self.terminator,
                                    checkpoint_semaphore, terminator_semaphore), daemon=False)
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

    def _run_checkpoint(self, checkpoint_semaphore):
        self._cleaning_tasks()
        checkpoint_semaphore.acquire()
        logger.debug('Saving Checkerpoint')
        self.save()

    def _run_reporter(self, task, task_process, reporter, searcher, terminator,
                      checkpoint_semaphore, terminator_semaphore):
        last_result = None
        while task_process.is_alive():
            reported_result = reporter.fetch()
            if 'done' in reported_result and reported_result['done'] is True:
                reporter.move_on()
                terminator_semaphore.release()
                terminator.on_task_complete(task, last_result)
                task_process.join()
                if checkpoint_semaphore is not None:
                    checkpoint_semaphore.release()
                break
            self.add_training_result(task.task_id, reported_result)
            if terminator.on_task_report(task, reported_result):
                last_result = reported_result
                reporter.move_on()
            else:
                last_result = reported_result
                last_result['terminated'] = True
                logger.debug('Removing task with ID {} due to low performance'.format(task.task_id))
                terminator_semaphore.release()
                terminator.on_task_remove(task)
                task_process.join()
                if checkpoint_semaphore is not None:
                    checkpoint_semaphore.release()
                break
        searcher.update(task.args['config'], last_result[self._reward_attr])

    def state_dict(self, destination=None):
        destination = super(DistributedHyperbandScheduler, self).state_dict(destination)
        destination['terminator'] = pickle.dumps(self.terminator)
        return destination

    def load_state_dict(self, state_dict):
        super(DistributedHyperbandScheduler, self).load_state_dict(state_dict)
        self.terminator = pickle.loads(state_dict['terminator'])
        logger.info('Loading Terminator State {}'.format(self.terminator))
