import copy
import math
import os
import warnings
from collections import OrderedDict, defaultdict

import cloudpickle as pkl
import matplotlib.pyplot as plt
import mxnet as mx
from PIL import Image
from mxnet.gluon.data.vision import transforms

from .metrics import get_metric_instance
from .nets import get_network
from .utils import *
from ..utils import collect_params, update_params
from autogluon.core import AutoGluonObject
from autogluon.core.utils import save, load, tqdm
from autogluon.core.task.base import BasePredictor


__all__ = ['Classifier']


class Classifier(BasePredictor):
    """Trained Image Classifier returned by fit() that can be used to make predictions on new images.
    Deprecated: please use autogluon.vision.ImagePredictor starting v0.1.0.

    Attributes
    ----------

    Examples
    --------
    >>> import autogluon.core as ag
    >>> from autogluon.vision import ImagePredictor
    >>> dataset = ImagePredictor.Dataset(train_path='data/train',
    >>>                        test_path='data/test')
    >>> classifier = ImagePredictor().fit(dataset,
    >>>                       nets=ag.space.Categorical['resnet18_v1', 'resnet34_v1'],
    >>>                       time_limit=time_limit,
    >>>                       ngpus_per_trial=1)
    >>> image = 'data/test/BabyShirt/BabyShirt_323.jpg'
    >>> ind, prob = classifier.predict(image)
    """

    def __init__(self, model, results, eval_func, scheduler_checkpoint,
                 args, ensemble=0, format_results=True, **kwargs):
        warnings.warn('Classifier is deprecated starting v0.1.0, please use `autogluon.vision.ImagePredictor`.')
        self.model = model
        self.eval_func = eval_func
        self.results = self._format_results(results) if format_results else results
        self.scheduler_checkpoint = scheduler_checkpoint
        self.args = args
        self.ensemble = ensemble

    @classmethod
    def load(cls, checkpoint):
        """Load trained Image Classifier from directory specified by `checkpoint`.
        """
        state_dict = load(checkpoint)
        args = state_dict['args']
        results = pkl.loads(state_dict['results'])
        eval_func = state_dict['eval_func']
        scheduler_checkpoint = state_dict['scheduler_checkpoint']
        model_params = state_dict['model_params']
        ensemble = state_dict['ensemble']

        if ensemble <= 1:
            model_args = copy.deepcopy(args)
            model_args.update(results['best_config'])
            model = get_network(args.net, num_classes=results['num_classes'], ctx=mx.cpu(0))
            update_params(model, model_params)
        else:
            raise NotImplemented
        return cls(model, results, eval_func, scheduler_checkpoint, args,
                   ensemble, format_results=False)

    def state_dict(self, destination=None):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        model_params = collect_params(self.model)
        destination['model_params'] = model_params
        destination['eval_func'] = self.eval_func
        destination['results'] = pkl.dumps(self.results)
        destination['scheduler_checkpoint'] = self.scheduler_checkpoint
        destination['args'] = self.args
        destination['ensemble'] = self.ensemble
        return destination

    def save(self, checkpoint):
        """ Save image classifier to folder specified by `checkpoint`.
        """
        state_dict = self.state_dict()
        save(state_dict, checkpoint)

    def predict(self, X, input_size=224, crop_ratio=0.875, set_prob_thresh=0.001, plot=False):
        """Predict class-index and associated class probability for each image in a given dataset (or just a single image). 
        
        Parameters
        ----------
        X : str or :class:`autogluon.vision.ImagePredictor.Dataset` or list of `autogluon.vision.ImagePredictor.Dataset`
            If str, should be path to the input image (when we just want to predict on single image).
            If class:`autogluon.vision.ImagePredictor.Dataset`, should be dataset of multiple images in same format as training dataset.
            If list of `autogluon.vision.ImagePredictor.Dataset`, should be a set of test dataset with different scales of origin images.
        input_size : int
            Size of the images (pixels).
        plot : bool
            Whether to plot the image being classified.
        set_prob_thresh: float
            Results with probability below threshold are set to 0 by default.

        Examples
        --------
        >>> import autogluon.core as ag
        >>> from autogluon.vision import ImagePredictor
        >>> train_data = ImagePredictor.Dataset(train_path='~/data/train')
        >>> classifier = ImagePredictor().fit(train_data,
        >>>                       nets=ag.space.Categorical['resnet18_v1', 'resnet34_v1'],
        >>>                       time_limit=600, ngpus_per_trial=1)
        >>> test_data = ImagePredictor.Dataset('~/data/test', train=False)
        >>> class_index, class_probability = classifier.predict('example.jpg')
        """

        input_size = self.model.input_size if hasattr(self.model, 'input_size') else input_size
        resize = int(math.ceil(input_size / crop_ratio))

        transform_size = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        def predict_img(img, ensemble=False):
            proba = self.predict_proba(img)
            if ensemble:
                return proba
            else:
                ind = mx.nd.argmax(proba, axis=1).astype('int')
                idx = mx.nd.stack(mx.nd.arange(proba.shape[0], ctx=proba.context), ind.astype('float32'))
                probai = mx.nd.gather_nd(proba, idx)
                return ind, probai, proba

        def avg_prediction(different_dataset, threshold=0.001):
            result = defaultdict(list)
            inds, probas, probals_all = [], [], []
            for i in range(len(different_dataset)):
                for j in range(len(different_dataset[0])):
                    result[j].append(different_dataset[i][j])

            for c in result.keys():
                proba_all = sum([*result[c]]) / len(different_dataset)
                proba_all = (proba_all >= threshold) * proba_all
                ind = mx.nd.argmax(proba_all, axis=1).astype('int')
                idx = mx.nd.stack(mx.nd.arange(proba_all.shape[0], ctx=proba_all.context), ind.astype('float32'))
                proba = mx.nd.gather_nd(proba_all, idx)
                inds.append(ind.asscalar())
                probas.append(proba.asnumpy())
                probals_all.append(proba_all.asnumpy().flatten())
            return inds, probas, probals_all

        def predict_imgs(X):
            if isinstance(X, list):
                different_dataset = []
                for i, x in enumerate(X):
                    proba_all_one_dataset = []
                    tbar = tqdm(range(len(x.items)))
                    for j, x_item in enumerate(x):
                        tbar.update(1)
                        proba_all = predict_img(x_item[0], ensemble=True)
                        tbar.set_description('ratio:[%d],The input picture [%d]' % (i, j))
                        proba_all_one_dataset.append(proba_all)
                    different_dataset.append(proba_all_one_dataset)
                inds, probas, probals_all = avg_prediction(different_dataset, threshold=set_prob_thresh)
            else:
                inds, probas, probals_all = [], [], []
                tbar = tqdm(range(len(X.items)))
                for i, x in enumerate(X):
                    tbar.update(1)
                    ind, proba, proba_all = predict_img(x[0])
                    tbar.set_description(
                        'The input picture [%d] is classified as [%d], with probability %.2f ' %
                        (i, ind.asscalar(), proba.asscalar())
                    )
                    inds.append(ind.asscalar())
                    probas.append(proba.asnumpy())
                    probals_all.append(proba_all.asnumpy().flatten())
            return inds, probas, probals_all

        if isinstance(X, str) and os.path.isfile(X):
            img = mx.image.imread(filename=X)
            if plot:
                plt.imshow(img.asnumpy())
                plt.show()

            img = transform_size(img)
            return predict_img(img)

        if isinstance(X, AutoGluonObject):
            X = X.init()
            return predict_imgs(X)

        if isinstance(X, list) and len(X) > 1:
            X_group = []
            for X_item in X:
                X_item = X_item.init()
                X_group.append(X_item)
            return predict_imgs(X_group)

    @staticmethod
    def loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def predict_proba(self, X):
        """Produces predicted class probabilities for a given image.
        """
        pred = self.model(X.expand_dims(0))
        return mx.nd.softmax(pred)

    def evaluate(self, dataset, input_size=224, ctx=[mx.cpu()]):
        """Evaluate predictive performance of trained image classifier using given test data.
        
        Parameters
        ----------
        dataset : :class:`autogluon.vision.ImagePredictor.Dataset`
            The dataset containing test images (must be in same format as the training dataset).
        input_size : int
            Size of the images (pixels).
        ctx : List of mxnet.context elements.
            Determines whether to use CPU or GPU(s), options include: `[mx.cpu()]` or `[mx.gpu()]`.
        
        Examples
        --------
        >>> import autogluon.core as ag
        >>> from autogluon.vision import ImagePredictor as vision
        >>> train_data = ImagePredictor.Dataset(train_path='~/data/train')
        >>> classifier = ImagePredictor().fit(train_data,
        >>>                       nets=ag.space.Categorical['resnet18_v1', 'resnet34_v1'],
        >>>                       time_limit=600, ngpus_per_trial=1)
        >>> test_data = ImagePredictor.Dataset('~/data/test', train=False)
        >>> test_acc = classifier.evaluate(test_data)
        """
        args = self.args
        net = self.model
        batch_size = args.batch_size * max(len(ctx), 1)
        metric = get_metric_instance(args.metric)
        input_size = net.input_size if hasattr(net, 'input_size') else input_size

        test_data, _, batch_fn, _ = get_data_loader(dataset, input_size, batch_size, args.num_workers, True, None)
        tbar = tqdm(test_data)
        for batch in tbar:
            self.eval_func(net, batch, batch_fn, metric, ctx)
            _, test_reward = metric.get()
            tbar.set_description('{}: {}'.format(args.metric, test_reward))
        _, test_reward = metric.get()
        return test_reward

    def evaluate_predictions(self, y_true, y_pred):
        raise NotImplementedError  # TODO
