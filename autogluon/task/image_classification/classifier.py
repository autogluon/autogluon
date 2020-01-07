import os
import math
import pickle
import numpy as np
from PIL import Image

import mxnet as mx
import matplotlib.pyplot as plt
from mxnet.gluon.data.vision import transforms

from ...core import AutoGluonObject
from .utils import *
from .metrics import get_metric_instance
from ..base.base_predictor import BasePredictor
from ...utils import save, load, tqdm
from ...utils.pil_transforms import *

__all__ = ['Classifier']

class Classifier(BasePredictor):
    """
    Trained Image Classifier returned by fit()
    """
    def __init__(self, model, results, eval_func, scheduler_checkpoint,
                 args, **kwargs):
        self.model = model
        self.eval_func = eval_func
        self.results = self._format_results(results)
        self.scheduler_checkpoint = scheduler_checkpoint
        self.args = args

    @classmethod
    def load(cls, checkpoint):
        state_dict = load(checkpoint)
        args = state_dict['args']
        results = state_dict['results']
        eval_func = pickle.loads(state_dict['eval_func'])
        scheduler_checkpoint = state_dict['scheduler_checkpoint']
        model_params = state_dict['model_params']

        model_args = copy.deepcopy(args)
        model_args.update(results['best_config'])
        model = get_network(args.net, args.dataset.num_classes)
        update_params(model, model_params)
        return cls(eval_func, model, eval_func, scheduler_checkpoint, args)

    def state_dict(self, destination=None):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination['model_params'] = collect_params(self.model)
        destination['eval_func'] = pickle.dumps(self.eval_func)
        destination['results'] = self.results
        destination['scheduler_checkpoint'] = self.scheduler_checkpoint
        destination['args'] = self.args
        return destination

    def save(self, checkpoint):
        save(self.state_dict(), checkpoint)

    def predict(self, X, input_size=224, plot=True):
        """ Predict class-index of a given image, as well as associated class probability.
        
        Parameters
        ----------
            X : str or :func:`autogluon.task.ImageClassification.Dataset`
                Path to the input image or dataset of multiple images.
        
        Examples
        --------
        >>> from autogluon import ImageClassification as task
        >>> train_data = task.Dataset(train_path='~/data/train')
        >>> classifier = task.fit(train_data,
        >>>                       nets=ag.space.Categorical['resnet18_v1', 'resnet34_v1'],
        >>>                       time_limits=600, ngpus_per_trial=1, num_trials=4)
        >>> test_data = task.Dataset('~/data/test', train=False)
        >>> class_index, class_probability = classifier.predict('example.jpg')
        """
        # model inference
        input_size = self.model.input_size if hasattr(self.model, 'input_size') else input_size
        resize = int(math.ceil(input_size / 0.875))
        transform_fn = Compose([
                Resize(resize),
                CenterCrop(input_size),
                ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        def predict_img(img):
            # load and display the image
            proba = self.predict_proba(img)
            ind = mx.nd.argmax(proba, axis=1).astype('int')
            idx = mx.nd.stack(mx.nd.arange(proba.shape[0], ctx=proba.context),
                              ind.astype('float32'))
            probai = mx.nd.gather_nd(proba, idx)
            return ind, probai
        if isinstance(X, str) and os.path.isfile(X):
            img = self.loader(X)
            if plot:
                plt.imshow(np.array(img))
                plt.show()
            img = transform_fn(img)
            return predict_img(img)
        if isinstance(X, AutoGluonObject):
            X = X.init()
        inds, probas = [], []
        for x in X:
            ind, proba = predict_img(x[0])
            inds.append(ind)
            probas.append(proba)
        return inds, probas

    @staticmethod
    def loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def predict_proba(self, X):
        """ Produces predicted class probabilities.
        """
        pred = self.model(X.expand_dims(0))
        return mx.nd.softmax(pred)

    def evaluate(self, dataset, input_size=224, ctx=[mx.cpu()]):
        """Evaluate this classifier's predictions on test data.
         Args:
            dataset: test dataset
        
        Examples
        --------
        >>> from autogluon import ImageClassification as task
        >>> train_data = task.Dataset(train_path='~/data/train')
        >>> classifier = task.fit(train_data,
        >>>                       nets=ag.space.Categorical['resnet18_v1', 'resnet34_v1'],
        >>>                       time_limits=600, ngpus_per_trial=1, num_trials = 4)
        >>> test_data = task.Dataset('~/data/test', train=False)
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
        raise NotImplementedError # TODO

    @staticmethod
    def _format_results(results): # TODO: remove since this has been moved to base_predictor.py
        def _merge_scheduler_history(training_history, config_history, reward_attr):
            trial_info = {}
            for tid, config in config_history.items():
                trial_info[tid] = {}
                trial_info[tid]['config'] = config
                if tid in training_history:
                    trial_info[tid]['history'] = training_history[tid]
                    trial_info[tid]['metadata'] = {}

                    if len(training_history[tid]) > 0 and reward_attr in training_history[tid][-1]:
                        last_history = training_history[tid][-1]
                        trial_info[tid][reward_attr] = last_history.pop(reward_attr)
                        trial_info[tid]['metadata'].update(last_history)
            return trial_info

        training_history = results.pop('training_history')
        config_history = results.pop('config_history')
        results['trial_info'] = _merge_scheduler_history(training_history, config_history,
                                                         results['reward_attr'])
        results[results['reward_attr']] = results['best_reward']
        results['search_space'] = results['metadata'].pop('search_space')
        results['search_strategy'] = results['metadata'].pop('search_strategy')
        return results
