#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:57:36 2020

@author: yirawan
"""

#vBERT-base/large, Roberta base/large, distillBERT
import logging
import matplotlib.pyplot as plt

import mxnet as mx
from mxnet import gluon, nd
import gluonnlp as nlp

from ...core import *
from ...searcher import *
from ...scheduler import *
from ...scheduler.resource import get_cpu_count, get_gpu_count
from ..base import BaseTask
from ...utils import update_params

from .network import BertForQA
from .dataset import get_dataset
from .pipeline import *
from .predictor import BERT_QA_Predictor

__all__ = ['FineTuneSQuADBERT']

logger = logging.getLogger(__name__)

class FineTuneSQuADBERT(BaseTask):
    """AutoGluon Task for finetune QA BERT based on their content
    """
    @staticmethod
    def Dataset(*args, **kwargs):
        """Dataset of text examples to make predictions for. 
           See :meth:`autogluon.task.TextClassification.get_dataset`
        """
        return get_dataset(*args, **kwargs)

    @staticmethod
    def fit(dataset='squad1.1',
            accumulate=4,
            output_dir='./output_dir',
            net=Categorical('bert_12_768_12'),
            dtype='float32',
            verbose=False,
            batch_size=32,
            dev_batch_size=32,
            lr=Real(2e-05, 2e-04, log=True),
            nthreads_per_trial=4,
            ngpus_per_trial=1,
            epsilon=1e-6,
            log_interval=100,
            optimizer='bertadam',
            warmup_ratio=0.01,
            version_2 = 'false',
            max_seq_length=128,
            doc_stride=128,
            max_query_length=64,
            n_best_size=20,
            max_answer_length=30,
            pretrained_dataset=Categorical('book_corpus_wiki_en_uncased',
                                           'openwebtext_book_corpus_wiki_en_uncased'),
            sentencepiece=None,
            only_predict=None,
            pretrained_bert_parameters=False,
            uncased=True,
            pretrained=False,
            null_score_diff_threshold=0.0,
            only_calibration=False,
            num_calib_batches=10,
            quantized_dtype='auto',
            calib_mode='customize',
            lr_scheduler='cosine',
            seed=0,
            epochs=3,
            early_stop=False,
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
            **kwargs):
                
        logger.warning('Warning: `if you use your own model, pls change the net-param` ')
        
        if auto_search:
            # The strategies can be injected here, for example: automatic suggest some hps
            # based on the dataset statistics
            pass

        nthreads_per_trial = get_cpu_count() if nthreads_per_trial > get_cpu_count() else nthreads_per_trial
        ngpus_per_trial = get_gpu_count() if ngpus_per_trial > get_gpu_count() else ngpus_per_trial

        Finetune_SQuAD_BERT.register_args(    
            dataset=dataset,
            accumulate=accumulate,
            output_dir=output_dir,
            pretrained_dataset=pretrained_dataset,
            net=net,
            dtype=dtype,
            lr=lr,
            warmup_ratio=warmup_ratio,
            early_stop=early_stop,
            max_seq_length=max_seq_length,
            log_interval=log_interval,
            epsilon=epsilon,
            seed=seed,
            lr_scheduler=lr_scheduler,
            ngpus_per_trial=ngpus_per_trial,
            batch_size=batch_size,
            dev_batch_size=dev_batch_size,
            epochs=epochs,
            num_workers=nthreads_per_trial,
            hybridize=hybridize,
            verbose=verbose,
            optimizer=optimizer,
            version_2 = version_2,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            n_best_size=n_best_size,
            max_answer_length=max_answer_length,
            sentencepiece=sentencepiece,
            only_predict=only_predict,
            pretrained_bert_parameters=pretrained_bert_parameters,
            uncased=uncased,
            pretrained=pretrained,
            null_score_diff_threshold=null_score_diff_threshold,
            only_calibration=only_calibration,
            num_calib_batches=num_calib_batches,
            quantized_dtype=quantized_dtype,
            calib_mode=calib_mode,
            final_fit=False,
            **kwargs)

        scheduler_options = {
            'resource': {'num_cpus': nthreads_per_trial, 'num_gpus': ngpus_per_trial},
            'checkpoint': checkpoint,
            'num_trials': num_trials,
            'time_out': time_limits,
            'resume': resume,
            'visualizer': visualizer,
            'time_attr': 'epoch',
            'reward_attr': 'accuracy',
            'dist_ip_addrs': dist_ip_addrs,
            'searcher': search_strategy,
            'search_options': search_options,
        }
        if search_strategy == 'hyperband':
            scheduler_options.update({
                'searcher': 'random',
                'max_t': epochs,
                'grace_period': grace_period if grace_period else epochs//4})
        results = BaseTask.run_fit(Finetune_SQuAD_BERT, search_strategy,
                                   scheduler_options)
        args = sample_config(Finetune_SQuAD_BERT.args, results['best_config'])
        get_model_params = results.pop('get_model_args')
        get_model_params['ctx'] = mx.cpu(0)
        bert, _ = nlp.model.get_model(**get_model_params)
        model = BertForQA(bert,prefix=None)
        update_params(model, results.pop('model_params'))
        transform = results.pop('transform')
        test_transform = results.pop('test_transform')
        # test(not completed)
        print("fit is finished, while predictor still in-progress")
        
        return results

                