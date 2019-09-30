import os
import pickle
import logging
import threading
import numpy as np
import multiprocessing as mp

from .scheduler import *
from ..scheduler.resource import Resources
from ..scheduler.reporter import StatusReporter
from ..scheduler.hyperband import Hyperband_Manager
from .fifo import FIFO_Scheduler

__all__ = ['Hyperband_Scheduler']

logger = logging.getLogger(__name__)


# Async version of Hyperband used in computation heavy tasks such as deep learning
class Hyperband_Scheduler(FIFO_Scheduler):
    """Implements the Async Hyperband
    This should provide similar theoretical performance as HyperBand but
    avoid straggler issues that HyperBand faces. One implementation detail
    is when using multiple brackets, task allocation to bracket is done
    randomly with over a softmax probability.
    See https://arxiv.org/abs/1810.05934

    Args:
        train_fn (callable): A task launch function for training. Note: please add the `@autogluon_method` decorater to the original function.
        args (object): Default arguments for launching train_fn.
        resource (dict): Computation resources. For example, `{'num_cpus':2, 'num_gpus':1}`
        searcher (object): Autogluon searcher. For example, autogluon.searcher.RandomSampling
        time_attr (str): A training result attr to use for comparing time. Note that you can pass in something non-temporal such as `training_epoch` as a measure of progress, the only requirement is that the attribute should increase monotonically.
        reward_attr (str): The training result objective value attribute. As with `time_attr`, this may refer to any objective value. Stopping procedures will use this attribute.
        max_t (float): max time units per task. Trials will be stopped after max_t time units (determined by time_attr) have passed.
        grace_period (float): Only stop tasks at least this old in time. The units are the same as the attribute named by `time_attr`.
        reduction_factor (float): Used to set halving rate and amount. This is simply a unit-less scalar.
        brackets (int): Number of brackets. Each bracket has a different halving rate, specified by the reduction factor.

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
        >>> myscheduler = Hyperband_Scheduler(train_fn, args,
        >>>                                   resource={'num_cpus': 2, 'num_gpus': 0},
        >>>                                   searcher=searcher, num_trials=20,
        >>>                                   reward_attr='accuracy',
        >>>                                   time_attr='epoch',
        >>>                                   grace_period=1)
    """

    def __init__(self, train_fn, args, resource, searcher,
                 checkpoint='./exp/checkerpoint.ag',
                 resume=False,
                 num_trials=None,
                 time_attr="training_epoch",
                 reward_attr="accuracy",
                 max_t=100, grace_period=10,
                 reduction_factor=4, brackets=1,
                 visualizer='none'):
        super(Hyperband_Scheduler, self).__init__(train_fn, args, resource, searcher,
                                                  checkpoint, resume, num_trials,
                                                  time_attr, reward_attr, visualizer)
        self.terminator = Hyperband_Manager(time_attr, reward_attr, max_t, grace_period,
                                            reduction_factor, brackets)

    def add_task(self, task):
        """Adding a training task to the scheduler.

        Args:
            task (:class:`autogluon.scheduler.Task`): a new trianing task
        """
        logger.debug("Adding A New Task {}".format(task))
        Hyperband_Scheduler.RESOURCE_MANAGER._request(task.resources)
        with self.LOCK:
            state_dict_path = os.path.join(os.path.dirname(self._checkpoint),
                                           'task{}_state_dict.ag'.format(task.task_id))
            reporter = StatusReporter(state_dict_path)
            task.args['reporter'] = reporter
            task.args['task_id'] = task.task_id
            task.args['resources'] = task.resources
            
            self.terminator.on_task_add(task)
            # main process
            tp = mp.Process(target=Hyperband_Scheduler._run_task, args=(
                task.fn, task.args, task.resources,
                Hyperband_Scheduler.RESOURCE_MANAGER))
            # reporter thread
            checkpoint_semaphore = mp.Semaphore(0) if self._checkpoint else None
            rp = threading.Thread(target=self._run_reporter,
                                  args=(task, tp, reporter, self.searcher, self.terminator,
                                        checkpoint_semaphore), daemon=False)
            tp.start()
            rp.start()
            task_dict = {'TASK_ID': task.task_id, 'Config': task.args['config'],
                         'Process': tp, 'ReporterThread': rp}
            # checkpoint thread
            if self._checkpoint is not None:
                sp = threading.Thread(target=self._run_checkpoint, args=(checkpoint_semaphore,),
                                      daemon=False)
                sp.start()
                task_dict['CheckpointThead'] = sp
            self.scheduled_tasks.append(task_dict)

    def _run_reporter(self, task, task_process, reporter, searcher, terminator,
                      checkpoint_semaphore):
        last_result = None
        while task_process.is_alive():
            reported_result = reporter.fetch()
            if 'done' in reported_result and reported_result['done'] is True:
                terminator.on_task_complete(task, last_result)
                reporter.move_on()
                task_process.join()
                if checkpoint_semaphore is not None:
                    checkpoint_semaphore.release()
                break
            self.add_training_result(task.task_id, reported_result)
            if terminator.on_task_report(task, reported_result):
                reporter.move_on()
            else:
                logger.debug('Removing task {} due to low performance'.format(task))
                last_result = reported_result
                last_result['terminated'] = True
                task_process.terminate()
                terminator.on_task_remove(task)
                task_process.join()
                Hyperband_Scheduler.RESOURCE_MANAGER._release(task.resources)
                if checkpoint_semaphore is not None:
                    checkpoint_semaphore.release()
                break
            last_result = reported_result
        searcher.update(task.args['config'], last_result[self._reward_attr])
        if searcher.is_best(task.args['config']):
            searcher.update_best_state(reporter.dict_path)

    def state_dict(self, destination=None):
        destination = super(Hyperband_Scheduler, self).state_dict(destination)
        destination['terminator'] = pickle.dumps(self.terminator)
        return destination

    def load_state_dict(self, state_dict):
        super(Hyperband_Scheduler, self).load_state_dict(state_dict)
        self.terminator = pickle.loads(state_dict['terminator'])
        logger.info('Loading Terminator State {}'.format(self.terminator))

    def __repr__(self):
        reprstr = self.__class__.__name__ + '(' +  \
            'terminator: ' + self.terminator
        return reprstr
