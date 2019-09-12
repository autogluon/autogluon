
import logging
import numpy as np

from mxnet import gluon, nd
from mxnet.gluon.data.vision import transforms
from gluoncv.data import transforms as gcv_transforms

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

__all__ = ['ImageClassification']

logger = logging.getLogger(__name__)


class ImageClassification(BaseTask):
    class Dataset(BaseTask.Dataset):
        """The image classification dataset.

        Args:
            name: the dataset name.
            train_path: the training data location
            val_path: the validation data location.
            batch_size: the batch size.
            num_workers: the number of workers used in DataLoader.
            transform_train_fn: the transformation function for training data.
            transform_val_fn: the transformation function for validation data.
            transform_train_list: the compose list of Transformations for training data.
            transform_val_list: the compose list of Transformations for validation data.
            batchify_train_fn: the batchify function defined for training data.
            batchify_val_fn: the batchify function defined for validation data.
        """
        def __init__(self, name=None, train_path=None, val_path=None, batch_size=64, num_workers=4,
                     transform_train_fn=None, transform_val_fn=None,
                     transform_train_list=[
                         transforms.Resize(480),
                         transforms.RandomResizedCrop(224),
                         transforms.RandomFlipLeftRight(),
                         transforms.RandomColorJitter(brightness=0.4,
                                                      contrast=0.4,
                                                      saturation=0.4),
                         transforms.RandomLighting(0.1),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
                     transform_val_list=[
                         transforms.Resize(256),
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
                     batchify_train_fn=None, batchify_val_fn=None,
                     **kwargs):
            super(ImageClassification.Dataset, self).__init__(name, train_path, val_path, batch_size, num_workers,
                                          transform_train_fn, transform_val_fn,
                                          transform_train_list, transform_val_list,
                                          batchify_train_fn, batchify_val_fn, **kwargs)
            self._split = None
            self._read_dataset(**kwargs)
            # TODO (cgraywang): add search space, handle batch_size, num_workers
            self._add_search_space()

        @property
        def split(self):
            return self._split

        @split.setter
        def split(self, value):
            self._split = value

        def _read_dataset(self, **kwargs):
            # TODO (cgraywang): put transform in the search space
            try:
                self.train = get_dataset(self.name, train=True)
                self.val = None
                self.test = get_dataset(self.name, train=False)
                self.num_classes = len(np.unique(self.train._label))
                # self.num_classes = DataAnalyzer.stat_dataset(self.train)[0]
                # DataAnalyzer.check_dataset(self.train, self.val)
            except ValueError:
                if self.train_path is not None:
                    if '.rec' not in self.train_path:
                        self.train = gluon.data.vision.ImageFolderDataset(self.train_path)
                        self.val = None
                        if 'test_path' in kwargs:
                            self.test = gluon.data.vision.ImageFolderDataset(kwargs['test_path'])
                        self.num_classes = len(np.unique([e[1] for e in self.train]))
                    elif '.rec' in self.train_path:
                        self.train = gluon.data.vision.ImageRecordDataset(self.train_path)
                        self.val = None
                        if 'test_path' in kwargs:
                            self.test = gluon.data.vision.ImageRecordDataset(kwargs['test_path'])
                        self.num_classes = len(np.unique([e[1] for e in self.train]))
                elif 'test_path' in kwargs:
                    if '.rec' not in kwargs['test_path']:
                        self.test = gluon.data.vision.ImageFolderDataset(kwargs['test_path'])
                    elif '.rec' in kwargs['test_path']:
                        self.test = gluon.data.vision.ImageRecordDataset(kwargs['test_path'])
                else:
                    raise NotImplementedError
            self.split = np.random.choice(range(1, 10), 1)[0]

        def _add_search_space(self):
            pass

        def _get_search_space_strs(self):
            pass

        def __repr__(self):
            repr_str = ""
            # try:
            #     train_stats = DataAnalyzer.stat_dataset(self.train)
            #     val_stats = DataAnalyzer.stat_dataset(self.val)
            #     repr_str = "AutoGluon Dataset: " \
            #                "\n ======== " \
            #                "\n name = %s" \
            #                "\n ======== " \
            #                "\n Train data statistic " \
            #                "\n number of classes = %d" \
            #                "\n number of samples = %d" \
            #                "\n mean (label) = %.2f" \
            #                "\n std (label) = %.2f" \
            #                "\n var (label) = %.2f" \
            #                "\n ======== " \
            #                "\n Val data statistic " \
            #                "\n number of classes = %d" \
            #                "\n number of samples = %d" \
            #                "\n mean (label) = %.2f" \
            #                "\n std (label) = %.2f" \
            #                "\n var (label) = %.2f" % (self.name,
            #                                           train_stats[0], train_stats[1],
            #                                           train_stats[2], train_stats[3],
            #                                           train_stats[4],
            #                                           val_stats[0], val_stats[1],
            #                                           val_stats[2], val_stats[3],
            #                                           val_stats[4])
            # except AttributeError:
            #     # TODO: add more info for folder dataset
            #     repr_str = "AutoGluon Dataset: " \
            #                "\n ======== " \
            #                "\n name = %s" \
            #                "\n ======== " % self.name
            return repr_str

    @staticmethod
    def fit(data,
            nets=Nets([
                get_model('resnet18_v1'),
                get_model('resnet34_v1')]),
            optimizers=Optimizers([
                get_optim('sgd'),
                get_optim('adam')]),
            metrics=Metrics([
                get_metric('Accuracy')]),
            losses=Losses([
                get_loss('SoftmaxCrossEntropyLoss')]),
            searcher='random',
            trial_scheduler='fifo',
            resume=False,
            savedir='checkpoint/exp1.ag',
            visualizer='tensorboard',
            stop_criterion={
                'time_limits': 24*60*60,
                'max_metric': 1.0,
                'num_trials': 100
            },
            resources_per_trial={
                'num_gpus': 1,
                'num_training_epochs': 100
            },
            backend='default',
            **kwargs):
        """
        Fit networks on dataset

        Args:
            data: Input data. task.Datasets.
            nets: autogluon.Nets.
            optimizers: autogluon.Optimizers.
            metrics: autogluon.Metrics.
            losses: autogluon.Losses.
            stop_criterion (dict): The stopping criteria.
            resources_per_trial (dict): Machine resources to allocate per trial.
            savedir (str): Local dir to save training results to.
            searcher: Search Algorithm.
            trial_scheduler: Scheduler for executing the experiment. Choose among FIFO (default) and HyperBand.
            resume (bool): If checkpoint exists, the experiment will resume from there.
            backend: support autogluon default backend.

        Example:
            >>> dataset = task.Dataset(name='shopeeiet', train_path='data/train',
            >>>                         test_path='data/test')
            >>> net_list = ['resnet18_v1', 'resnet34_v1']
            >>> nets = ag.Nets(net_list)
            >>> adam_opt = ag.optims.Adam(lr=ag.space.Log('lr', 10 ** -4, 10 ** -1),
            >>>                           wd=ag.space.Log('wd', 10 ** -6, 10 ** -2))
            >>> sgd_opt = ag.optims.SGD(lr=ag.space.Log('lr', 10 ** -4, 10 ** -1),
            >>>                         momentum=ag.space.Linear('momentum', 0.85, 0.95),
            >>>                         wd=ag.space.Log('wd', 10 ** -6, 10 ** -2))
            >>> optimizers = ag.Optimizers([adam_opt, sgd_opt])
            >>> searcher = 'random'
            >>> trial_scheduler = 'fifo'
            >>> savedir = 'checkpoint/demo.ag'
            >>> resume = False
            >>> time_limits = 3*60
            >>> max_metric = 1.0
            >>> num_trials = 4
            >>> stop_criterion = {
            >>>       'time_limits': time_limits,
            >>>       'max_metric': max_metric,
            >>>       'num_trials': num_trials
            >>> }
            >>> num_gpus = 1
            >>> num_training_epochs = 2
            >>> resources_per_trial = {
            >>>       'num_gpus': num_gpus,
            >>>       'num_training_epochs': num_training_epochs
            >>> }
            >>> results = task.fit(dataset,
            >>>                     nets,
            >>>                     optimizers,
            >>>                     searcher=searcher,
            >>>                     trial_scheduler=trial_scheduler,
            >>>                     resume=resume,
            >>>                     savedir=savedir,
            >>>                     stop_criterion=stop_criterion,
            >>>                     resources_per_trial=resources_per_trial)
        """
        return BaseTask.fit(data, nets, optimizers, metrics, losses, searcher, trial_scheduler,
                            resume, savedir, visualizer, stop_criterion, resources_per_trial,
                            backend,
                            reward_attr='accuracy',
                            train_func=train_image_classification,
                            **kwargs)
