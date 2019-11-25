import os
import math
import pickle
import copy
from collections import OrderedDict
import mxnet as mx
import matplotlib.pyplot as plt

from .classification_models import *
from ...utils import *
from .pipeline import *
from .metrics import get_metric_instance
from ..base.base_predictor import BasePredictor
if in_ipynb():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

__all__ = ['TextClassificationPredictor']

class TextClassificationPredictor(BasePredictor):
    """
    Classifier returned by task.fit()

    Example user workflow:
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
        model = get_network(args.net)
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

    def predict(self, X):
        """The task predict function given an input.
         Args:
            sentence: the input
         Example:
            >>> ind = predictor.predict('this is cool')
        """
        proba = self.predict_proba(X)
        ind = mx.nd.argmax(proba, axis=1).astype('int')
        return ind

    def predict_proba(self, X):
        """The task predict probability function given an input.
         Args:
            sentence: the input
         Example:
            >>> prob = predictor.predict_proba('this is cool')
        """
        pred = self.model(X.expand_dims(0))
        return mx.nd.softmax(pred)

    def evaluate(self, dataset, ctx=[mx.cpu()], *args):
        """The task evaluation function given the test dataset.
         Args:
            dataset: test dataset
         Example:
            >>> from autogluon import TextClassification as task
            >>> dataset = task.Dataset(test_path='~/data/test')
            >>> test_reward = predictor.evaluate(dataset)
        """
        args = self.args
        net = self.model
        batch_size = args.batch_size * max(len(ctx), 1)
        metric = get_metric_instance(args.metric)
        _, dev_data_list, _, _ = preprocess_data(
            args.bert_tokenizer, args.task, batch_size, args.dev_batch_size, args.max_len, args.vocabulary, args.pad)
        tbar = tqdm(enumerate(dev_data_list))
        for i, batch in tbar:
            self.eval_func(net, batch, metric, ctx)
            _, test_reward = metric.get()
            tbar.set_description('{}: {}'.format(args.metric, test_reward))
        _, test_reward = metric.get()
        return test_reward

    def _save_model(self, *args, **kwargs):
        raise NotImplemented

    def evaluate_predictions(self, y_true, y_pred):
        raise NotImplemented

    @staticmethod
    def _format_results(results):
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
        results[results['reward_attr']] = results.pop('best_reward')
        results['search_space'] = results['metadata'].pop('search_space')
        results['search_strategy'] = results['metadata'].pop('search_strategy')
        return results
