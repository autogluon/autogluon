import os
import pickle
import logging
import threading
import multiprocessing as mp

from .scheduler import *
from .resource_manager import Resources
from .reporter import StatusReporter
from ..basic import save, load
from ..utils import mkdir

__all__ = ['FIFO_Scheduler']

logger = logging.getLogger(__name__)

class FIFO_Scheduler(TaskScheduler):
    """Simple scheduler that just runs trials in submission order.
    Args:
        train_fn (callable): A task launch function for training.
            Note: please add the `@autogluon_method` decorater to the original function.
        args (object): Default arguments for launching train_fn.
        resource (dict): Computation resources.
            For example, `{'num_cpus':2, 'num_gpus':1}`
        searcher (object): Autogluon searcher.
            For example, autogluon.searcher.RandomSampling
        reward_attr (str): The training result objective value attribute. As
            with `time_attr`, this may refer to any objective value. Stopping
            procedures will use this attribute.
    """
    def __init__(self, train_fn, args, resource, searcher, checkpoint=None,
                 resume=False, num_trials=None, reward_attr='accuracy'):
        self.train_fn = train_fn
        self.args = args
        self.resource = resource
        self.searcher = searcher
        self.num_trials = num_trials
        self._checkpoint = checkpoint
        self._reward_attr = reward_attr
        if resume:
            if os.path.isfile(checkpoint):
                self.load_state_dict(load(checkpoint))
            else:
                msg = 'checkpoint path {} is not available for resume.'.format(checkpoint)
                logger.exception(msg)
                raise FileExistsError(msg)

    def run(self, num_trials=None):
        """Run multiple number of trials
        """
        self.num_trials = num_trials if num_trials else self.num_trials
        logger.info('Starting Experiments')
        logger.info('Num of Finished Tasks is {}'.format(self.num_finished_tasks))
        logger.info('Num of Pending Tasks is {}'.format(self.num_trials - self.num_finished_tasks))
        for i in range(self.num_finished_tasks, self.num_trials):
            self.schedule_next()
        self.join_tasks()

    def save(self, checkpoint=None):
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
                    Resources(**self.resource))
        self.add_task(task)

    def add_task(self, task):
        # adding the task
        logger.debug("Adding A New Task {}".format(task))
        FIFO_Scheduler.RESOURCE_MANAGER._request(task.resources)
        with self.LOCK:
            if task.resources.num_gpus > 0:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(task.resources.gpu_ids)[1:-1]
            reporter = StatusReporter()
            task.args['reporter'] = reporter
            tp = mp.Process(target=FIFO_Scheduler._run_task, args=(
                            task.fn, task.args, task.resources,
                            FIFO_Scheduler.RESOURCE_MANAGER))
            checkpoint_semaphore = mp.Semaphore(0) if self._checkpoint else None
            rp = threading.Thread(target=FIFO_Scheduler._run_reporter, args=(task, tp, reporter,
                                  self.searcher, self._reward_attr, checkpoint_semaphore))
            tp.start()
            rp.start()
            if self._checkpoint is not None:
                sp = threading.Thread(target=self._run_checkpoint, args=(checkpoint_semaphore,))
                sp.start()
            self.SCHEDULED_TASKS.append({'TASK_ID': task.task_id, 'Config': task.args['config'],
                                         'Process': tp, 'ReporterProcess': rp})

    def _run_checkpoint(self, checkpoint_semaphore):
        checkpoint_semaphore.acquire()
        logger.debug('Saving Checkerpoint')
        self.save()

    @staticmethod
    def _run_reporter(task, task_process, reporter, searcher, reward_attr,
                      checkpoint_semaphore):
        last_result = None
        while task_process.is_alive():
            kwargs = reporter.fetch()
            if 'done' in kwargs and kwargs['done'] is True:
                task_process.join()
                if checkpoint_semaphore is not None:
                    checkpoint_semaphore.release()
                break
            reporter.move_on()
            last_result = kwargs
        searcher.update(task.args['config'], last_result[reward_attr])
        #reporting = [last_result[reward_attr]]
        #if 'model_params' in last_result:
        #    # update model params if reported
        #    reporting.append(last_result['model_params'])
        #searcher.update(task.args['config'], *reporting)

    def get_best_config(self):
        FIFO_Scheduler.join_tasks()
        return self.searcher.get_best_config()

    def get_best_reward(self):
        FIFO_Scheduler.join_tasks()
        return self.searcher.get_best_reward()

    def state_dict(self, destination=None):
        destination = super(FIFO_Scheduler, self).state_dict(destination)
        destination['num_trials'] = self.num_trials
        #destination['searcher_state_dict'] = self.searcher.state_dict()
        destination['searcher'] = pickle.dumps(self.searcher)
        return destination

    def load_state_dict(self, state_dict):
        super(FIFO_Scheduler, self).load_state_dict(state_dict)
        #self.searcher.load_state_dict(state_dict['searcher_state_dict'])
        self.searcher = pickle.loads(state_dict['searcher'])
        logger.debug('Loading Searcher State {}'.format(self.searcher))
