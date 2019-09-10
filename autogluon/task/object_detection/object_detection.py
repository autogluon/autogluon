import logging

from .model_zoo import *
from .losses import *
from .metrics import *
from ...network import Nets
from ...optimizer import Optimizers, get_optim
from ...loss import Losses
from ...metric import Metrics
from ...utils.data_analyzer import DataAnalyzer
from ..base import *
from .dataset import *
from .pipeline import *

__all__ = ['ObjectDetection']

logger = logging.getLogger(__name__)


class ObjectDetection(BaseTask):
    class Dataset(BaseTask.Dataset):
        def __init__(self, name=None, train_path=None, val_path=None, batch_size=32, num_workers=4,
                     transform_train_fn='SSDDefaultTrainTransform',
                     transform_val_fn='SSDDefaultValTransform',
                     transform_train_list=None, transform_val_list=None,
                     batchify_train_fn=batchify_fn(), batchify_val_fn=batchify_val_fn(),
                     **kwargs):
            super(ObjectDetection.Dataset, self).__init__(name, train_path, val_path, batch_size, num_workers,
                                          transform_train_fn, transform_val_fn,
                                          transform_train_list, transform_val_list,
                                          batchify_train_fn, batchify_val_fn, **kwargs)
            self._read_dataset(**kwargs)
            # TODO (cgraywang): add search space, handle batch_size, num_workers
            self._add_search_space()

        def _read_dataset(self, **kwargs):
            # TODO (cgraywang): put transform in the search space
            try:
                # Only support VOC in the defined dataset list
                self.train = get_dataset(self.name, splits=[(2007, 'trainval'), (2012, 'trainval')])
                self.val = get_dataset(self.name, splits=[(2007, 'test')])
                self.num_classes = len(self.train.classes)
                # TODO (cgraywang): add DataAnalyzer
                # self.num_classes = DataAnalyzer.stat_dataset(self.train)[0]
                # DataAnalyzer.check_dataset(self.train, self.val)
            except ValueError:
                raise NotImplementedError

        def _add_search_space(self):
            pass

        def _get_search_space_strs(self):
            pass

        def __repr__(self):
            try:
                train_stats = DataAnalyzer.stat_dataset(self.train)
                val_stats = DataAnalyzer.stat_dataset(self.val)
                repr_str = "AutoGluon Dataset: " \
                           "\n ======== " \
                           "\n name = %s" \
                           "\n ======== " \
                           "\n Train data statistic " \
                           "\n number of classes = %d" \
                           "\n number of samples = %d" \
                           "\n mean (label) = %.2f" \
                           "\n std (label) = %.2f" \
                           "\n var (label) = %.2f" \
                           "\n ======== " \
                           "\n Val data statistic " \
                           "\n number of classes = %d" \
                           "\n number of samples = %d" \
                           "\n mean (label) = %.2f" \
                           "\n std (label) = %.2f" \
                           "\n var (label) = %.2f" % (self.name,
                                                      train_stats[0], train_stats[1],
                                                      train_stats[2], train_stats[3],
                                                      train_stats[4],
                                                      val_stats[0], val_stats[1],
                                                      val_stats[2], val_stats[3],
                                                      val_stats[4])
            except AttributeError:
                # TODO: add more info for folder dataset
                repr_str = "AutoGluon Dataset: " \
                           "\n ======== " \
                           "\n name = %s" \
                           "\n ======== " % self.name
            return repr_str

    @staticmethod
    def fit(data,
            nets=Nets([
                get_model('ssd_512_resnet50_v1_coco'),
                get_model('faster_rcnn_fpn_resnet101_v1d_coco'),
                get_model('yolo3_mobilenet1.0_coco')]),
            optimizers=Optimizers([
                get_optim('sgd'),
                get_optim('adam')]),
            metrics=Metrics([
                get_metric('VOC07MApMetric')]),
            losses=Losses([
                get_loss('SSDMultiBoxLoss')]),
            searcher=None,
            trial_scheduler=None,
            resume=False,
            savedir='checkpoint/exp1.ag',
            visualizer='tensorboard',
            stop_criterion={
                'time_limits': 1 * 60 * 60,
                'max_metric': 1,
                'max_trial_count': 2
            },
            resources_per_trial={
                'max_num_gpus': 0,
                'max_num_cpus': 4,
                'max_training_epochs': 3
            },
            backend='default',
            **kwargs):
        r"""
        Fit networks on dataset

        Parameters
        ----------
        data: Input data. It could be:
            autogluon.Datasets
            task.Datasets
        nets: autogluon.Nets
        optimizers: autogluon.Optimizers
        metrics: autogluon.Metrics
        losses: autogluon.Losses
        stop_criterion (dict): The stopping criteria. The keys may be any field in
            the return result of 'train()', whichever is reached first.
            Defaults to empty dict.
        resources_per_trial (dict): Machine resources to allocate per trial,
            e.g. ``{"max_num_cpus": 64, "max_num_gpus": 8}``. Note that GPUs will not be
            assigned unless you specify them here.
        savedir (str): Local dir to save training results to.
        searcher: Search Algorithm.
        trial_scheduler: Scheduler for executing
            the experiment. Choose among FIFO (default) and HyperBand.
        resume (bool): If checkpoint exists, the experiment will
            resume from there.
        backend: support autogluon default backend, ray. (Will support SageMaker)
        **kwargs: Used for backwards compatibility.

        Returns
        ----------
        results:
            model: the parameters associated with the best model. (TODO:)
            metric: validation set metric
            config: best configuration
            time: total time cost
        """
        return BaseTask.fit(data, nets, optimizers, metrics, losses, searcher, trial_scheduler,
                            resume, savedir, visualizer, stop_criterion, resources_per_trial,
                            backend,
                            reward_attr='map',
                            train_func=train_object_detection,
                            **kwargs)
