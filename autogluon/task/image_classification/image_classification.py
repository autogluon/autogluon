import logging

from mxnet import gluon, nd
from mxnet.gluon.data.vision import transforms
from gluoncv.data import transforms as gcv_transforms

from ..optimizer import *
from ..basic.space import *
from ..searcher import *
from ..scheduler import *

from .nets import *
from .dataset import get_built_in_dataset
from .pipeline import train_image_classification

from ...utils import EasyDict as ezdict

__all__ = ['ImageClassification']

logger = logging.getLogger(__name__)

class ImageClassification(object):
    @staticmethod
    def fit(train_dataset='cifar10',
            val_dataset='cifar10',
            net=ListSpace('CIFAR_ResNet20_v1', 'CIFAR_ResNet20_v2'),
            # training hps
            optimizer=SGD(learning_rate=LogLinearSpace(1e-4, 1e-2),
                          momentum=LogLinearSpace(0.85, 0.95),
                          wd=LogLinearSpace(1e-5, 1e-3)),
            loss=gluon.loss.SoftmaxCrossEntropyLoss(),
            batch_size=64,
            epochs=20,
            metric='accuracy',
            num_workers=4,
            num_cpus=4,
            num_gpus=0,
            # searcher and scheduler options
            searcher=RandomSampling,
            scheduler=DistributedFIFOScheduler,
            resume=False,
            checkpoint='checkpoint/exp1.ag',
            visualizer='none',
            num_trials=2,
            dist_ip_addrs=[],
            grace_period=None):

        # Any strategy can be injected here, for example: automatic suggest some hps
        # based on the data layout
        train_image_classification.update(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            net=net,
            optimizer=optimizer,
            loss=loss,
            metric=metric,
            num_workers=num_workers,
            num_gpus=num_gpus,
            batch_size=batch_size,
            epochs=epochs)

        args = train_image_classification.args
        cs = train_image_classification.cs

        searcher_inst = searcher(cs)
        scheduler_inst = scheduler(train_image_classification, args,
                                   resource={'num_cpus': num_cpus, 'num_gpus': num_gpus},
                                   searcher=searcher_inst,
                                   checkpoint=checkpoint,
                                   num_trials=num_trials,
                                   resume=resume,
                                   time_attr='epoch',
                                   reward_attr=metric,
                                   dist_ip_addrs=dist_ip_addrs,
                                   max_t=args.epochs,
                                   grace_period=grace_period if grace_period else args.epochs//4)
        scheduler_inst.run()
        scheduler_inst.join_tasks()
        ImageClassification.scheduler = scheduler_inst
        return ezdict({'best_config':scheduler_inst.get_best_config(),
                       'best_reward':scheduler_inst.get_best_reward()})

    def shut_down(self):
        ImageClassification.scheduler.shutdown()
