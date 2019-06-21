from typing import AnyStr

from mxboard import SummaryWriter

from ..estimator import *

__name__ = 'MXBoardHandler'

logger = logging.getLogger(__name__)

__all__ = ['MXBoardHandler']


class MXBoardHandler(TrainBegin, TrainEnd, EpochBegin, EpochEnd, BatchBegin, BatchEnd):

    def __init__(self, task_id: int, log_dir: AnyStr = './logs', checkpoint_dir: AnyStr = None,
                 events_to_log: set = None, when_to_log: set = None):
        self.task_id = task_id
        logger.info(log_dir)
        self.log_dir = log_dir
        self.summary_writer = SummaryWriter(logdir=log_dir, flush_secs=2)
        self.checkpoint_dir = checkpoint_dir
        self.events_to_log = None
        self.when_to_log = None
        self.is_graph_added = False
        self.current_epoch = 0

    def train_begin(self, estimator, *args, **kwargs):
        # TODO Ideally the add_graph should be called here, but in Gluon it requires one to have called forward pass
        #  atleast once on the graph. So this callback might not be that useful.
        pass

    def train_end(self, estimator, *args, **kwargs):
        if self.checkpoint_dir:
            with SummaryWriter(logdir=self.log_dir) as sw:
                sw.export_scalars('{}.json'.format(self.checkpoint_dir))

    def epoch_begin(self, estimator, *args, **kwargs):
        pass

    def epoch_end(self, estimator, *args, **kwargs):
        # Get the metrics from estimator.
        # Add those as scalars.
        # TODO Add metrics as per the list specified to track. Currently just doing Validation metrics
        with SummaryWriter(logdir=self.log_dir, flush_secs=2) as sw:
            for metric in estimator.val_metrics:
                sw.add_scalar(tag=metric.name, value=('task_%d' % self.task_id, metric.get()[1]),
                              global_step=self.current_epoch)

        self.current_epoch += 1

    def batch_begin(self, estimator, *args, **kwargs):
        pass

    def batch_end(self, estimator, *args, **kwargs):
        pass
