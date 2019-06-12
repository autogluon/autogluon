import json
import multiprocessing as mp

from ..estimator import *

logger = logging.getLogger(__name__)

class StatusReporter(EpochEnd):
    """Report status through the training scheduler.
    Example:
        >>> def train_func(config, reporter):
        >>>     assert isinstance(reporter, StatusReporter)
        >>>     reporter(timesteps_this_iter=1)
    """

    def __init__(self, task_id: int):#, result_queue, continue_semaphore):
        self._queue = mp.Queue(1)
        self._last_report_time = None
        self._continue_semaphore = mp.Semaphore(1)
        self._last_report_time = time.time()
        self.current_epoch = 0
        self.task_id = task_id

    def __call__(self, **kwargs):
        """Report updated training status.
        Pass in `done=True` when the training job is completed.
        Args:
            kwargs: Latest training result status.
        Example:
            >>> reporter(accuracy=1, training_iters=4)
        """
        report_time = time.time()
        if 'time_this_iter' not in kwargs:
            kwargs['time_this_iter'] = report_time - self._last_report_time
        self._last_report_time = report_time

        self._queue.put(kwargs.copy(), block=True)
        self._continue_semaphore.acquire()

        logger.info('StatusReporter reporting: {}'.format(json.dumps(kwargs)))

    def fetch(self, block=True):
        kwargs = self._queue.get(block=block)
        return kwargs

    def move_on(self):
        self._continue_semaphore.release()

    def _start(self):
        """Adjust the real starting time
        """
        self._last_report_time = time.time()

    def epoch_end(self, estimator: Estimator, *args, **kwargs):
        self.current_epoch += 1
        # TODO (shaabhn): fix for custom eval metric
        self(task_id=self.task_id, epoch=self.current_epoch, accuracy=estimator.val_metrics[0].get()[1])
