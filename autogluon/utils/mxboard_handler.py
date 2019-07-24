from typing import AnyStr

from mxboard import SummaryWriter

from ..estimator import *

__name__ = 'MXBoardHandler'

logger = logging.getLogger(__name__)

__all__ = ['MXBoardHandler']


class MXBoardHandler(TrainBegin, TrainEnd, EpochBegin, EpochEnd, BatchBegin, BatchEnd):

    # TODO : Improve this with PubSub so that this runs and logs on a separate thread
    def __init__(self, task_id: int, log_dir: AnyStr = './logs', checkpoint_dir: AnyStr = None,
                 events_to_log: set = None, when_to_log: set = None, trial_args=None):
        self.task_id = task_id
        logger.info(log_dir)
        self.log_dir = log_dir
        self.summary_writer = SummaryWriter(logdir=log_dir, flush_secs=2)
        self.checkpoint_dir = checkpoint_dir
        self.events_to_log = None
        self.when_to_log = None
        self.is_graph_added = False
        self.current_epoch = 0
        self.batch_index = 0
        self.trial_args = trial_args
        self.time_to_train = 0

    def train_begin(self, estimator, *args, **kwargs):
        # TODO Ideally the add_graph should be called here, but in Gluon it requires one to have called forward pass
        #  atleast once on the graph. So this callback might not be that useful.
        if self.trial_args:
            with SummaryWriter(logdir=self.log_dir) as sw:
                text = ''
                for key in self.trial_args:
                    str = '{} : {}\n'.format(key, self.trial_args[key])
                    text += str
                sw.add_text(tag='hyperparams_selected', text=text, global_step=self.trial_args['task_id'])
            logger.info('Hyperparams Selected : {}'.format(text))

        # Start the timer.
        self.time_to_train = time.time()

    def train_end(self, estimator, *args, **kwargs):

        # Log the timer.
        self.time_to_train = time.time() - self.time_to_train
        with SummaryWriter(logdir=self.log_dir, flush_secs=2) as sw:
            sw.add_scalar(tag='train_time_taken', value=('task_%d' % self.task_id, self.time_to_train), global_step=0)
            # TODO : What should global step be here ?
        if self.checkpoint_dir:
            with SummaryWriter(logdir=self.log_dir, flush_secs=2) as sw:
                sw.export_scalars('{}.json'.format(self.checkpoint_dir))

    def epoch_begin(self, estimator, *args, **kwargs):
        pass

    def epoch_end(self, estimator, *args, **kwargs):
        # Get the metrics from estimator.
        # Add those as scalars.
        # TODO Add metrics as per the list specified to track. Currently just doing Validation metrics
        with SummaryWriter(logdir=self.log_dir, flush_secs=2) as sw:
            for metric in estimator.val_metrics:
                sw.add_scalar(tag='{}_epoch'.format(metric.name), value=('task_%d' % self.task_id, metric.get()[1]),
                              global_step=self.current_epoch)

        self.current_epoch += 1
        self.batch_index = 0

    def batch_begin(self, estimator, *args, **kwargs):
        pass

    def batch_end(self, estimator, *args, **kwargs):
        if self.batch_index != 0 and self.batch_index % 100 == 0:
            with SummaryWriter(logdir=self.log_dir, flush_secs=2) as sw:
                for metric in estimator.train_metrics:
                    sw.add_scalar(tag='{}_batch'.format(metric.name),
                                  value=('task_%d_epoch_%d' % (self.task_id, self.current_epoch), metric.get()[1]),
                                  global_step=self.batch_index)

        self.batch_index += 1
