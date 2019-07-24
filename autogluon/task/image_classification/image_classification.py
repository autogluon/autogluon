import logging

from mxnet import gluon, nd
from mxnet.gluon.data.vision import transforms
from gluoncv.data import transforms as gcv_transforms

from .model_zoo import *
from .losses import *
from .metrics import *
from ...network import Nets
from ...optim import Optimizers, get_optim
from ...loss import Losses
from ...metric import Metrics
from ...utils.data_analyzer import DataAnalyzer
from ..base import *
from .dataset import *
from .pipeline import *

__all__ = ['ImageClassification']

logger = logging.getLogger(__name__)


class ImageClassification(BaseTask):
    class Dataset(BaseTask.Dataset):
        def __init__(self, name=None, train_path=None, val_path=None, batch_size=64, num_workers=4,
                     transform_train_fn=None, transform_val_fn=None,
                     transform_train_list=[
                         gcv_transforms.RandomCrop(32, pad=4),
                         transforms.RandomFlipLeftRight(),
                         transforms.ToTensor(),
                         transforms.Normalize([0.4914, 0.4822, 0.4465],
                                              [0.2023, 0.1994, 0.2010])
                     ],
                     transform_val_list=[
                         transforms.ToTensor(),
                         transforms.Normalize([0.4914, 0.4822, 0.4465],
                                              [0.2023, 0.1994, 0.2010])
                     ],
                     batchify_train_fn=None, batchify_val_fn=None,
                     **kwargs):
            super(ImageClassification.Dataset, self).__init__(name, train_path, val_path, batch_size, num_workers,
                                          transform_train_fn, transform_val_fn,
                                          transform_train_list, transform_val_list,
                                          batchify_train_fn, batchify_val_fn, **kwargs)
            self._read_dataset(**kwargs)
            # TODO (cgraywang): add search space, handle batch_size, num_workers
            self._add_search_space()

        def _read_dataset(self, **kwargs):
            # TODO (cgraywang): put transform in the search space
            try:
                self.train = get_dataset(self.name, train=True)
                self.val = get_dataset(self.name, train=False)
                self.num_classes = DataAnalyzer.stat_dataset(self.train)[0]
                DataAnalyzer.check_dataset(self.train, self.val)
            except ValueError:
                self.train = gluon.data.vision.ImageFolderDataset(self.train_path)
                self.val = gluon.data.vision.ImageFolderDataset(self.val_path)
                self.num_classes = kwargs['num_classes']

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
                get_model('cifar_resnet20_v1'),
                get_model('cifar_resnet56_v1'),
                get_model('cifar_resnet110_v1')]),
            optimizers=Optimizers([
                get_optim('sgd'),
                get_optim('adam')]),
            metrics=Metrics([
                get_metric('Accuracy')]),
            losses=Losses([
                get_loss('SoftmaxCrossEntropyLoss')]),
            searcher=None,
            trial_scheduler=None,
            resume=False,
            savedir='checkpoint/exp1.ag',
            visualizer='tensorboard',
            stop_criterion={
                'time_limits': 1 * 60 * 60,
                'max_metric': 1.0,
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
            val_accuracy: validation set accuracy
            config: best configuration
            time: total time cost
        """
        return BaseTask.fit(data, nets, optimizers, metrics, losses, searcher, trial_scheduler,
                            resume, savedir, visualizer, stop_criterion, resources_per_trial,
                            backend,
                            reward_attr='accuracy',
                            train_func=train_image_classification,
                            **kwargs)
