import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon.data.vision import transforms
from gluoncv.data import transforms as gcv_transforms

from ...core.optimizer import *
from ...core import *
from ...searcher import *
from ...scheduler import *
from ...scheduler.resource import get_cpu_count, get_gpu_count
from ..base import BaseTask

from .dataset import ImageClassificationDataset, get_built_in_dataset
from .pipeline import train_image_classification
from .metrics import get_metric_instance

__all__ = ['ImageClassification']

logger = logging.getLogger(__name__)

class ImageClassification(BaseTask):
    """AutoGluon ImageClassification Task
    """
    Dataset = ImageClassificationDataset
    @staticmethod
    def fit(dataset,
            net=Categorical('ResNet34_v1b', 'ResNet50_v1b'),
            optimizer=Categorical(
                SGD(learning_rate=LogLinear(1e-4, 1e-2),
                    momentum=LogLinear(0.85, 0.95),
                    wd=LogLinear(1e-5, 1e-3)),
                Adam(learning_rate=LogLinear(1e-4, 1e-2),
                     wd=LogLinear(1e-5, 1e-3)),
            ),
            lr_scheduler='cosine',
            loss=gluon.loss.SoftmaxCrossEntropyLoss(),
            batch_size=64,
            input_size=224,
            epochs=20,
            metric='accuracy',
            num_cpus=get_cpu_count(),
            num_gpus=get_gpu_count(),
            hybridize=True,
            search_strategy='random',
            search_options={},
            time_limits=None,
            resume=False,
            checkpoint='checkpoint/exp1.ag',
            visualizer='none',
            num_trials=2,
            dist_ip_addrs=[],
            grace_period=None,
            auto_search=True,
            debug=False):

        """
        Auto fit on image classification dataset

        Args:
            dataset (str or autogluon.task.ImageClassification.Dataset): Training dataset.
            net (str, autogluon.AutoGluonObject, or ag.Categorical of AutoGluonObject): Network candidates.
            optimizer (str, autogluon.AutoGluonObject, or ag.Categorical of AutoGluonObject): optimizer candidates.
            metric (str or object): observation metric.
            loss (object): training loss function.
            num_trials (int): number of trials in the experiment.
            time_limits (int): training time limits in seconds.
            resources_per_trial (dict): Machine resources to allocate per trial.
            savedir (str): Local dir to save training results to.
            search_strategy (str): Search Algorithms ('random', 'bayesopt' and 'hyperband')
            resume (bool): If checkpoint exists, the experiment will resume from there.


        Example:
            >>> dataset = task.Dataset(train_path='~/data/train',
            >>>                        test_path='data/test')
            >>> results = task.fit(dataset,
            >>>                    nets=ag.Categorical['resnet18_v1', 'resnet34_v1'],
            >>>                    time_limits=time_limits,
            >>>                    num_gpus=1,
            >>>                    num_trials = 4)
        """
        if auto_search:
            # The strategies can be injected here, for example: automatic suggest some hps
            # based on the dataset statistics
            pass

        num_cpus = get_cpu_count() if num_cpus > get_cpu_count() else num_cpus
        num_gpus = get_gpu_count() if num_gpus > get_gpu_count() else num_gpus

        train_image_classification.update(
            dataset=dataset,
            net=net,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss=loss,
            metric=metric,
            num_gpus=num_gpus,
            batch_size=batch_size,
            input_size=input_size,
            epochs=epochs,
            num_workers=num_cpus,
            hybridize=hybridize,
            final_fit=False)

        scheduler_options = {
            'resource': {'num_cpus': num_cpus, 'num_gpus': num_gpus},
            'checkpoint': checkpoint,
            'num_trials': num_trials,
            'time_out': time_limits,
            'resume': resume,
            'visualizer': visualizer,
            'time_attr': 'epoch',
            'reward_attr': 'reward',
            'dist_ip_addrs': dist_ip_addrs,
            'searcher': search_strategy,
            'search_options': search_options,
            'debug': debug,
        }
        if search_strategy == 'hyperband':
            scheduler_options.update({
                'max_t': epochs,
                'grace_period': grace_period if grace_period else epochs//4})

        return BaseTask.run_fit(train_image_classification, search_strategy, scheduler_options)

    @classmethod
    def predict(cls, img):
        """The task predict function given an input.
         Args:
            img: the input
         Example:
            >>> ind, prob = task.predict('example.jpg')
        """
        # load and display the image
        img = mx.image.imread(img)
        plt.imshow(img.asnumpy())
        plt.show()
        # model inference
        args = cls.scheduler.train_fn.args
        dataset = args.dataset
        if isinstance(dataset, AutoGluonObject):
            dataset = dataset._lazy_init()
        transform_fn = dataset.transform_val
        img = transform_fn(img)
        ctx = mx.gpu(0)if args.num_gpus > 0 else mx.cpu()
        pred = cls.results.model(img.expand_dims(0).as_in_context(ctx))
        ind = mx.nd.argmax(pred, axis=1).astype('int')
        mx.nd.waitall()
        return ind, mx.nd.softmax(pred)[0][ind]

    @classmethod
    def evaluate(cls, dataset, model=None, input_size=224):
        """The task evaluation function given the test dataset.
         Args:
            dataset: test dataset
         Example:
            >>> from autogluon import ImageClassification as task
            >>> dataset = task.Dataset(name='shopeeiet', test_path='~/data/test')
            >>> test_reward = task.evaluate(dataset)
        """
        args = cls.scheduler.train_fn.args if hasattr(cls, 'scheduler') else train_image_classification.args
        batch_size = args.batch_size * max(args.num_gpus, 1)

        metric = get_metric_instance(args.metric)
        ctx = [mx.gpu(i) for i in range(args.num_gpus)] if args.num_gpus > 0 else [mx.cpu()]
        if isinstance(model, str):
            from ...nas.model_zoo import get_model
            model = get_model(model, pretrained=True, ctx=ctx)
            model.collect_params().reset_ctx(ctx)
        else:
            cls.results.model.collect_params().reset_ctx(ctx)
            model = cls.results.model

        if isinstance(dataset, AutoGluonObject):
            dataset = dataset.init()
        elif isinstance(dataset, str):
            dataset = get_built_in_dataset(dataset, train=False, input_size=input_size,
                                           batch_size=args.batch_size,
                                           num_workers=args.num_cpus).init()

        if isinstance(dataset, gluon.data.Dataset):
            test_data = gluon.data.DataLoader(
                dataset, batch_size=batch_size,
                shuffle=False, num_workers=args.num_workers)
        elif hasattr(dataset, 'test'):
            test_data = gluon.data.DataLoader(
                dataset.test, batch_size=batch_size, shuffle=False,
                num_workers=args.num_workers)
        else:
            test_data = dataset

        tbar = tqdm(enumerate(test_data))
        for i, batch in tbar:
            if isinstance(test_data, gluon.data.DataLoader):
                data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
                label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            else:
                data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
                label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = [model(X) for X in data]
            metric.update(label, outputs)
            _, test_reward = metric.get()
            tbar.set_description('{}: {}'.format(args.metric, test_reward))
        _, test_reward = metric.get()
        return test_reward
