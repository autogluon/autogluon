import os
import math
import pickle
import copy
from collections import OrderedDict
import mxnet as mx
import matplotlib.pyplot as plt

from ...utils import *
from .network import *
from .pipeline import *
from ..image_classification.classifier import Classifier
from ...core import AutoGluonObject

__all__ = ['TextClassificationPredictor']

class TextClassificationPredictor(Classifier):
    """
    Classifier returned by task.fit()

    Example user workflow:
    """
    def __init__(self, model, transform, test_transform,
                 results, scheduler_checkpoint, args):
        self.model = model
        self.transform = transform
        self.test_transform = test_transform
        self.results = self._format_results(results)
        self.scheduler_checkpoint = scheduler_checkpoint
        self.args = args

    def predict(self, X):
        """The task predict function given an input.
         Args:
            sentence: the input
         Example:
            >>> ind = predictor.predict('this is cool')
        """
        X = self.test_transform(X)
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

    def evaluate(self, dataset, ctx=[mx.cpu()]):
        """The task evaluation function given the test dataset.
         Args:
            dataset: test dataset
         Example:
            >>> from autogluon import TextClassification as task
            >>> dataset = task.Dataset(test_path='~/data/test')
            >>> test_reward = predictor.evaluate(dataset)
        """
        if isinstance(dataset, AutoGluonObject):
            dataset = dataset.init()

        args = self.args
        net = self.model
        batch_size = args.batch_size * max(len(ctx), 1)
        _, dev_data_list, _ = preprocess_data(
            args.bert_tokenizer, args.task, batch_size, args.dev_batch_size, args.max_len, args.vocabulary, args.pad)
        tbar = tqdm(enumerate(dev_data_list))
        for i, batch in tbar:
            eval_func(net, batch, metric, ctx[0], args)
            _, test_reward = metric.get()
            tbar.set_description('{}: {}'.format(args.metric, test_reward))
        _, test_reward = metric.get()
        return test_reward

def eval_func(model, loader_dev, metric, ctx, args):
    """Evaluate the model on validation dataset."""
    use_roberta = 'roberta' in args.bert_model
    metric.reset()
    for batch_id, seqs in enumerate(loader_dev):
        input_ids, valid_length, segment_ids, label = seqs
        input_ids = input_ids.as_in_context(ctx)
        valid_length = valid_length.as_in_context(ctx).astype('float32')
        label = label.as_in_context(ctx)
        if use_roberta:
            out = model(input_ids, valid_length)
        else:
            out = model(input_ids, segment_ids.as_in_context(ctx), valid_length)
        metric.update([label], [out])

    metric_nm, metric_val = metric.get()
    if not isinstance(metric_nm, list):
        metric_nm, metric_val = [metric_nm], [metric_val]
    mx.nd.waitall()
    return metric_nm, metric_val
