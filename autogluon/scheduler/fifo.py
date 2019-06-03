import os
import pickle
import json
import logging
import threading
import multiprocessing as mp
from collections import OrderedDict
from mxboard import SummaryWriter

from .scheduler import *
from ..resource import Resources
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
                 resume=False, num_trials=None, time_attr='epoch', reward_attr='accuracy',
                 visualizer='tensorboard'):
        super(FIFO_Scheduler, self).__init__()
        self.train_fn = train_fn
        self.args = args
        self.resource = resource
        self.searcher = searcher
        self.num_trials = num_trials
        self._checkpoint = checkpoint
        self._time_attr = time_attr
        self._reward_attr = reward_attr
        assert visualizer.lower() == 'tensorboard' or visualizer.lower() == 'mxboard', \
            'Only Tensorboard and MXboard are supported.'
        self.visualizer = SummaryWriter(
            logdir=os.path.join(os.path.splitext(checkpoint)[0], 'logs'),
            flush_secs=3,
            verbose=False)
        self.log_lock = mp.Lock()
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
        self.visualizer.export_scalars('{}.json'.format(os.path.splitext(self._checkpoint)[0]))
        self.visualizer.close()

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
            reporter = StatusReporter()
            task.args['reporter'] = reporter
            # main process
            tp = mp.Process(target=FIFO_Scheduler._run_task, args=(
                            task.fn, task.args, task.resources,
                            FIFO_Scheduler.RESOURCE_MANAGER))
            checkpoint_semaphore = mp.Semaphore(0) if self._checkpoint else None
            # reporter thread
            rp = threading.Thread(target=self._run_reporter, args=(task, tp, reporter,
                                  self.searcher, checkpoint_semaphore), daemon=False)
            tp.start()
            rp.start()
            # checkpoint thread
            if self._checkpoint is not None:
                sp = threading.Thread(target=self._run_checkpoint, args=(checkpoint_semaphore,),
                                      daemon=False)
                sp.start()
            self.scheduled_tasks.append({'TASK_ID': task.task_id, 'Config': task.args['config'],
                                         'Process': tp, 'ReporterProcess': rp})

    def _cleaning_tasks(self):
        with self.LOCK:
            for i, task_dick in enumerate(self.scheduled_tasks):
                if not task_dick['Process'].is_alive():
                    task_dict = self.scheduled_tasks.pop(i)
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
                task_process.join()
                if checkpoint_semaphore is not None:
                    checkpoint_semaphore.release()
                break
            self.visualizer.add_scalar(tag='loss',
                                       value=('task %d valid_loss' % task.task_id,
                                              reported_result['loss']),
                                       global_step=reported_result['epoch'])
            self.visualizer.add_scalar(tag='accuracy_curves',
                                       value=('task %d valid_acc' % task.task_id,
                                              reported_result[self._reward_attr]),
                                       global_step=reported_result['epoch'])
            reporter.move_on()
            last_result = reported_result
        searcher.update(task.args['config'], last_result[self._reward_attr])

    def get_best_config(self):
        self.join_tasks()
        return self.searcher.get_best_config()

    def get_best_reward(self):
        self.join_tasks()
        return self.searcher.get_best_reward()

    def state_dict(self, destination=None):
        destination = super(FIFO_Scheduler, self).state_dict(destination)
        destination['searcher'] = pickle.dumps(self.searcher)
        return destination

    def load_state_dict(self, state_dict):
        super(FIFO_Scheduler, self).load_state_dict(state_dict)
        self.searcher = pickle.loads(state_dict['searcher'])
        logger.debug('Loading Searcher State {}'.format(self.searcher))
