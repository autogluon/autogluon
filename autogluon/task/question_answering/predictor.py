#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:56:43 2020

@author: yirawan
"""
from collections import namedtuple, OrderedDict
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

__all__ = ['BERT_QA_Predictor']


class BERT_QA_Predictor(Classifier):
    """Trained Text Classifier returned by `fit()` that can be used to make predictions on new text data.
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
        """Predict class-index of a given sentence / text-snippet.
        
        Parameters
        ----------
        X : str
            The input sentence we should classify.
    
        Examples
        --------
        >>> class_index = predictor.predict('this is cool')
    
        Returns
        -------
        Int corresponding to index of the predicted class.
        """
        proba = self.predict_proba(X)
        ind = mx.nd.argmax(proba, axis=1).astype('int')
        return ind

#    def predict_proba(self, X):
#        """Predict class-probabilities of a given sentence / text-snippet.
#        
#        Parameters
#        ----------
#        X : str
#            The input sentence we should classify.
#        
#        Examples
#        --------
#        >>> class_probs = predictor.predict_proba('this is cool')
#        
#        Returns
#        -------
#        `mxnet.NDArray` containing predicted probabilities of each class.
#        """
#        inputs = self.test_transform(X)
#        X, valid_length, segment_id = [mx.nd.array(np.expand_dims(x, 0)) for x in inputs]
#        pred = self.model(X, segment_id, valid_length)
#        return mx.nd.softmax(pred)

    def evaluate(self, dataset, ctx=[mx.cpu()]):
        """Evaluate predictive performance of trained text classifier using given test data.
        
        Parameters
        ----------
        dataset : :class:`autogluon.task.FineTuneSQuADBERT.Dataset`
            The dataset containing test sentences (must be in same format as the training dataset provided to fit).
        ctx : List of `mxnet.context` elements.
            Determines whether to use CPU or GPU(s), options include: `[mx.cpu()]` or `[mx.gpu()]`.
        
         Examples
         --------
        >>> from autogluon import FineTuneSQuADBERT as task
        >>> dataset = task.Dataset(test_path='~/data/test')
        >>> test_performance = predictor.evaluate(dataset)
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
            nlp.data.batchify.Stack(),
            nlp.data.batchify.Pad(axis=0, pad_val=vocab[pad_val], round_to=args.round_to),
            nlp.data.batchify.Pad(axis=0, pad_val=vocab[pad_val], round_to=args.round_to),
            nlp.data.batchify.Stack('float32'),
            nlp.data.batchify.Stack('float32'),
            nlp.data.batchify.Stack('float32'))
        dev_dataloader = mx.gluon.data.DataLoader(
                                    dataset=data_dev,
                                    batchify_fn=batchify_fn,
                                    num_workers=args.num_workers, 
                                    batch_size=args.dev_batch_size,
                                    shuffle=False,
                                    last_batch='keep')
                
    
        eval_func(net, dev_dataloader, metric, ctx)
        _, test_reward = metric.get()
         still tuning yet
        return True

def eval_func(model, loader_dev, metric, ctx):
    """Evaluate the model on validation dataset."""
    metric.reset()
    for batch_id, seqs in enumerate(loader_dev):
        input_ids, valid_length, segment_ids, label = seqs
        input_ids = input_ids.as_in_context(ctx)
        valid_length = valid_length.as_in_context(ctx).astype('float32')

        out = model(input_ids, segment_ids.as_in_context(ctx), valid_length)
        metric.update([label], [out])

    metric_nm, metric_val = metric.get()
    if not isinstance(metric_nm, list):
        metric_nm, metric_val = [metric_nm], [metric_val]
    mx.nd.waitall()
    return metric_nm, metric_val

