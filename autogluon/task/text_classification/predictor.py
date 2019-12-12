import os
import math
import pickle
import copy
import numpy as np
from collections import OrderedDict
import mxnet as mx
import matplotlib.pyplot as plt
import gluonnlp as nlp

from ...utils import *
from .network import *
from .pipeline import *
from .dataset import *
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
        self.use_roberta = 'roberta' in args.net
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
        inputs = self.test_transform(X)
        X, valid_length, segment_id = [mx.nd.array(np.expand_dims(x, 0)) for x in inputs]
        if self.use_roberta:
            pred = self.model(X, valid_length)
        else:
            pred = self.model(X, segment_id, valid_length)
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
        args = self.args
        net = self.model
        if isinstance(dataset, AutoGluonObject):
            dataset = dataset.init()
        if isinstance(dataset, AbstractGlueTask):
            dataset = dataset.get_dataset('dev')
        if isinstance(ctx, list):
            ctx = ctx[0]

        metric = mx.metric.Accuracy()
        dataset = dataset.transform(self.transform)
        vocab = self.transform.vocab
        pad_val = vocab[vocab.padding_token]
        batchify_fn = nlp.data.batchify.Tuple(
            nlp.data.batchify.Pad(axis=0, pad_val=pad_val),  # input
            nlp.data.batchify.Stack(),  # length
            nlp.data.batchify.Pad(axis=0, pad_val=0),  # segment
            nlp.data.batchify.Stack('int32'))  # label
        loader_dev = mx.gluon.data.DataLoader(
            dataset,
            batch_size=args.dev_batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            batchify_fn=batchify_fn)

        eval_func(net, loader_dev, metric, ctx, self.use_roberta)
        _, test_reward = metric.get()
        return test_reward

def eval_func(model, loader_dev, metric, ctx, use_roberta):
    """Evaluate the model on validation dataset."""
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
