#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:56:52 2020

@author: yirawan
"""
import os
import time
import itertools
import collections
import multiprocessing as mp
from functools import partial
import mxnet as mx
import numpy as np
import random
import warnings
import json
import io
from gluonnlp.data.bert.squad import convert_squad_examples
import gluonnlp as nlp
from gluonnlp.data import BERTTokenizer
import logging
from .dataset import *
from .transforms import BERTDatasetTransform
from ...core import *
from ...utils.mxutils import collect_params
from .network import BertForQALoss, BertForQA
from .network import get_F1_EM, predict, PredResult

log = logging.getLogger('gluonnlp')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    fmt='%(levelname)s:%(name)s:%(asctime)s %(message)s', datefmt='%H:%M:%S')


__all__ = ['Finetune_SQuAD_BERT', 'preprocess_data']
   

def preprocess_data(tokenizer, task, batch_size, dev_batch_size, max_seq_length=384,
                    vocab=None,doc_stride=128, max_query_length=64,lower=True,
                    num_workers=4,dtype='float32', pad=False, round_to=None,
                    is_training=True,sentencepiece=None,comm_backend='mxnet'):

    if comm_backend == 'horovod':
        import horovod.mxnet as hvd
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()
    else:
        rank = 0
        size = 1
    
    if dtype == 'float16':
        from mxnet.contrib import amp
        amp.init()
        
    """Loads a dataset into features"""
    pool = mp.Pool(num_workers)
    if sentencepiece:
        tokenizer = nlp.data.BERTSPTokenizer(sentencepiece, vocab, lower=lower)
    else:
        tokenizer = nlp.data.BERTTokenizer(vocab=vocab, lower=lower)
    pad_val = vocab[vocab.padding_token]
    vocab = tokenizer.vocab if vocab is None else vocab
    
    trans = BERTDatasetTransform(
                            tokenizer=tokenizer,
                            cls_token=vocab.cls_token,
                            sep_token=vocab.sep_token,
                            vocab=vocab,
                            max_seq_length=max_seq_length,
                            doc_stride=doc_stride,
                            max_query_length=max_query_length)
    # data train
    # task.dataset_train returns (segment_name, dataset)
    train_tsv = task.dataset_train()[1]
    example_trans = partial(convert_squad_examples,
                            is_training=is_training)
        # convert the raw dataset into raw features
    examples = pool.map(example_trans, train_tsv)
    data_train = mx.gluon.data.SimpleDataset(
            list(itertools.chain.from_iterable(pool.map(trans, examples))))
                    
    batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Stack(),
        nlp.data.batchify.Pad(axis=0, pad_val=vocab[pad_val], round_to=round_to),
        nlp.data.batchify.Pad(axis=0, pad_val=vocab[pad_val], round_to=round_to),
        nlp.data.batchify.Stack('float32'),
        nlp.data.batchify.Stack('float32'),
        nlp.data.batchify.Stack('float32'))
     
    """data dev"""
    # as same as above
    dev_tsv = task.dataset_dev()[1]
    example_trans = partial(convert_squad_examples,
                            is_training=is_training)
    examples = pool.map(example_trans, dev_tsv)
    raw_features = pool.map(trans, examples)
    data_dev = mx.gluon.data.SimpleDataset(
                list(itertools.chain.from_iterable(raw_features)))
        
    batchify_fn_calib = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0, pad_val=pad_val, round_to=round_to),
        nlp.data.batchify.Pad(axis=0, pad_val=pad_val, round_to=round_to),
        nlp.data.batchify.Stack('float32'),
        nlp.data.batchify.Stack('float32'))
    
    data_train_len = len(data_train)

    """calibration function on the dev dataset."""
    # task.dataset_train returns (segment_name, dataset)
    data_calib = data_dev.transform(lambda *example: (
        example[7],  # inputs_id
        example[9],  # segment_ids
        example[3],  # valid_length,
        example[10]))  # start_position,
    
    calib_dataloader = mx.gluon.data.DataLoader(
                                dataset=data_calib,
                                batchify_fn=batchify_fn_calib,
                                num_workers=4, 
                                batch_size=dev_batch_size,
                                shuffle=False, 
                                last_batch='keep')
    
    dev_dataloader = mx.gluon.data.DataLoader(
                                dataset=data_dev,
                                batchify_fn=batchify_fn,
                                num_workers=4, 
                                batch_size=dev_batch_size,
                                shuffle=False,
                                last_batch='keep')
                
    
    data_train = data_train.transform(lambda *example: (
        example[0],  # example_id
        example[7],  # inputs_id
        example[9],  # segment_ids
        example[3],  # valid_length,
        example[10],  # start_position,
        example[11]))  # end_position
    
    '''batch_sampler'''
    log.info('The number of examples after preprocessing:{}'.format(
        data_train_len))
    sampler = nlp.data.SplitSampler(data_train_len, num_parts=size,
                                    part_index=rank, even_size=True)
    num_train_examples = len(sampler)
    train_dataloader = mx.gluon.data.DataLoader(dataset=data_train,
                                                batchify_fn=batchify_fn,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                sampler=sampler)


    dev_dataset = mx.gluon.data.SimpleDataset(list(raw_features))
        
    
    pool.close()
    log.info('Number of records in dev data:{}'.format(len(dev_dataset)))

    return train_dataloader, calib_dataloader, dev_dataloader, dev_dataset,num_train_examples

    


@args()
def Finetune_SQuAD_BERT(args, reporter=None):

    logger = logging.getLogger(__name__)
    accumulate = args.accumulate
    output_dir = args.output_dir
    batch_size = args.batch_size
    if args.verbose:
        logger.setLevel(logging.INFO)
        logger.info(args)
    # data type with mixed precision training
    if args.dtype == 'float16':
        try:
            from mxnet.contrib import amp  # pylint: disable=ungrouped-imports
            # monkey patch amp list since topk does not support fp16
            amp.lists.symbol.FP32_FUNCS.append('topk')
            amp.lists.symbol.FP16_FP32_FUNCS.remove('topk')
            amp.init()
        except ValueError:
            # topk is already in the FP32_FUNCS list
            amp.init()
        except ImportError:
            # amp is not available
            logger.info('Mixed precision training with float16 requires MXNet >= '
                         '1.5.0b20190627. Please consider upgrading your MXNet version.')
            exit()
    if accumulate:
        logger.info('Using gradient accumulation. Effective batch size = ' \
                     'batch_size * accumulate = %d', accumulate * batch_size)

    
    # Step 1: add scripts every function and python objects in the original training script except for the training function
    # at the beginning of the decorated function
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
        
    dev_batch_size = args.dev_batch_size
    lr = args.lr
    # TODO support for multi-GPU
    ctx = [mx.gpu(i) for i in range(args.num_gpus)][0] if args.num_gpus > 0 else [mx.cpu()][0]
    log_interval = args.log_interval * accumulate if accumulate else args.log_interval

    # random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    mx.random.seed(args.seed)

    # some hyper-params
    optimizer = args.optimizer
    warmup_ratio = args.warmup_ratio
    version_2 = args.version_2
    null_score_diff_threshold = args.null_score_diff_threshold
    max_seq_length = args.max_seq_length
    doc_stride = args.doc_stride
    max_query_length = args.max_query_length
    n_best_size = args.n_best_size
    max_answer_length = args.max_answer_length
    
    dataset_name = args.pretrained_dataset
    task = args.dataset

    if max_seq_length <= max_query_length + 3:
        raise ValueError('The max_seq_length (%d) must be greater than max_query_length '
                         '(%d) + 3' % (max_seq_length, max_query_length))
    
    # vocabulary and tokenizer
    if args.sentencepiece:
        logging.info('loading vocab file from sentence piece model: %s', args.sentencepiece)
        if dataset_name:
            warnings.warn('Both --dataset_name and --sentencepiece are provided. '
                          'The vocabulary will be loaded based on --sentencepiece.')
        vocab = nlp.vocab.BERTVocab.from_sentencepiece(args.sentencepiece)
        dataset_name = None
    else:
        vocab = None
        
    # model and loss
    model_name = args.net
    dataset = args.pretrained_dataset
    only_predict = args.only_predict
    model_parameters = args.model_parameters
    pretrained_bert_parameters = args.pretrained_bert_parameters
    if pretrained_bert_parameters and model_parameters:
        raise ValueError('Cannot provide both pre-trained BERT parameters and '
                         'BertForQA model parameters.')
    lower = args.uncased
    get_model_params = {
        'name': model_name,
        'dataset_name': dataset,
        'pretrained': True,
        'ctx': ctx,
        'use_decoder': False,
        'use_classifier': False,
    }
    bert, vocabulary = nlp.model.get_model(**get_model_params)    
    pretrained = False
    # pretrained = not model_parameters and not pretrained_bert_parameters and not args.sentencepiece
    bert, vocab = nlp.model.get_model(
                                name=model_name,
                                dataset_name=dataset_name,
                                vocab=vocab,
                                pretrained=pretrained,
                                ctx=ctx,
                                use_pooler=False,
                                use_decoder=False,
                                use_classifier=False)
    net = BertForQA(bert=bert)
    net.hybridize(static_alloc=True)
        
        
    if args.sentencepiece:
        tokenizer = nlp.data.BERTSPTokenizer(args.sentencepiece, vocab, lower=lower)
    else:
        tokenizer = nlp.data.BERTTokenizer(vocab=vocab, lower=lower)

    
    if model_parameters:
        # load complete BertForQA parameters
        nlp.utils.load_parameters(net, model_parameters, ctx=ctx, cast_dtype=True)
    elif pretrained_bert_parameters:
        # only load BertModel parameters
        nlp.utils.load_parameters(bert, pretrained_bert_parameters, ctx=ctx,
                                  ignore_extra=True, cast_dtype=True)
        net.span_classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
    elif pretrained:
        # only load BertModel parameters
        net.span_classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
    else:
        # no checkpoint is loaded
        net.initialize(init=mx.init.Normal(0.02), ctx=ctx)
    
    loss_function = BertForQALoss()
    loss_function.hybridize(static_alloc=True)
    
    # calibration config
    only_calibration = args.only_calibration
    num_calib_batches = args.num_calib_batches
    quantized_dtype = args.quantized_dtype
    calib_mode = args.calib_mode


#    version = task.version
#    metrics = task.metrics
    mode = task.mode
#    debug = task.debug
    num_workers = args.num_workers
    """Training function."""
    # initialize classifier
    # data processing
    log.info('Loading %s data...', mode)
    train_dataloader, calib_dataloader, dev_dataloader, \
                            dev_dataset, num_examples = preprocess_data(
                                        tokenizer=tokenizer, 
                                        task=task,
                                        num_workers=num_workers,
                                        batch_size=batch_size, 
                                        dev_batch_size=dev_batch_size,
                                        max_seq_length=max_seq_length,
                                        doc_stride=doc_stride,
                                        max_query_length=max_query_length)
    log.info('Start Training')
    optimizer_params = {'learning_rate': lr, 'wd': 0.01}
    param_dict = net.collect_params()
    if args.comm_backend == 'horovod':
        import horovod.mxnet as hvd
        trainer = hvd.DistributedTrainer(param_dict, optimizer, optimizer_params)
    else:
        trainer = mx.gluon.Trainer(param_dict, optimizer, optimizer_params,
                                   update_on_kvstore=False)
    if args.dtype == 'float16':
        amp.init_trainer(trainer)
        
    step_size = batch_size * accumulate if accumulate else batch_size
    num_train_steps = int(num_examples / step_size * args.epochs)
    if args.training_steps:
        num_train_steps = args.training_steps

    num_warmup_steps = int(num_train_steps * warmup_ratio)
    def set_new_lr(step_num, batch_id):
        """set new learning rate"""
        # set grad to zero for gradient accumulation
        if accumulate:
            if batch_id % accumulate == 0:
                step_num += 1
        else:
            step_num += 1
        # learning rate schedule
        # Notice that this learning rate scheduler is adapted from traditional linear learning
        # rate scheduler where step_num >= num_warmup_steps, new_lr = 1 - step_num/num_train_steps
        if step_num < num_warmup_steps:
            new_lr = lr * step_num / num_warmup_steps
        else:
            offset = (step_num - num_warmup_steps) * lr / \
                (num_train_steps - num_warmup_steps)
            new_lr = lr - offset
        trainer.set_learning_rate(new_lr)
        return step_num
    
    # Do not apply weight decay on LayerNorm and bias terms
    for _, v in net.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    # Collect differentiable parameters
    params = [p for p in param_dict.values() if p.grad_req != 'null']

    # Set grad_req if gradient accumulation is required
    if accumulate:
        for p in params:
            p.grad_req = 'add'
    net.collect_params().zero_grad()

    epoch_tic = time.time()
    total_num = 0
    log_num = 0
    batch_id = 0
    step_loss = 0.0
    tic = time.time()
    step_num = 0

    tic = time.time()
    while step_num < num_train_steps:
        for _, data in enumerate(train_dataloader):
            # set new lr
            step_num = set_new_lr(step_num, batch_id)
            # forward and backward
            _, inputs, token_types, valid_length, start_label, end_label = data
            num_labels = len(inputs)
            log_num += num_labels
            total_num += num_labels

            with mx.autograd.record():
                out = net(inputs.as_in_context(ctx),
                          token_types.as_in_context(ctx),
                          valid_length.as_in_context(ctx).astype('float32'))

                loss = loss_function(out, [
                    start_label.as_in_context(ctx).astype('float32'),
                    end_label.as_in_context(ctx).astype('float32')
                ]).sum() / num_labels

                if accumulate:
                    loss = loss / accumulate
                if args.dtype == 'float16':
                    with amp.scale_loss(loss, trainer) as l:
                        mx.autograd.backward(l)
                        norm_clip = 1.0  * trainer._amp_loss_scaler.loss_scale
                else:
                    mx.autograd.backward(loss)
                    norm_clip = 1.0 

            # update
            if not accumulate or (batch_id + 1) % accumulate == 0:
                trainer.allreduce_grads()
                nlp.utils.clip_grad_global_norm(params, norm_clip)
                trainer.update(1)
                if accumulate:
                    param_dict.zero_grad()

            if args.comm_backend == 'horovod':
                step_loss += hvd.allreduce(loss, average=True).asscalar()
            else:
                step_loss += loss.asscalar()

            if (batch_id + 1) % log_interval == 0:
                toc = time.time()
                log.info('Batch: {}/{}, Loss={:.4f}, lr={:.7f} '
                         'Thoughput={:.2f} samples/s'
                         .format(batch_id % len(train_dataloader),
                                 len(train_dataloader), step_loss / log_interval,
                                 trainer.learning_rate, log_num/(toc - tic)))
                tic = time.time()
                step_loss = 0.0
                log_num = 0

            if step_num >= num_train_steps:
                break
            batch_id += 1

        log.info('Finish training step: %d', step_num)
        epoch_toc = time.time()
        log.info('Time cost={:.2f} s, Thoughput={:.2f} samples/s'.format(
            epoch_toc - epoch_tic, total_num / (epoch_toc - epoch_tic)))
        
        
    def calibration(net, num_calib_batches, quantized_dtype, calib_mode):
        """calibration function on the dev dataset."""
        assert ctx == mx.cpu(), \
            'Currently only supports CPU with MKL-DNN backend.'
        log.info('Now we are doing calibration on dev with %s.', ctx)
        collector = BertLayerCollector(clip_min=-50, clip_max=10, logger=log)
        num_calib_examples = dev_batch_size * num_calib_batches
        net = mx.contrib.quantization.quantize_net_v2(net, quantized_dtype=quantized_dtype,
                                                      exclude_layers=[],
                                                      quantize_mode='smart',
                                                      quantize_granularity='channel-wise',
                                                      calib_data=dev_dataloader,
                                                      calib_mode=calib_mode,
                                                      num_calib_examples=num_calib_examples,
                                                      ctx=ctx,
                                                      LayerOutputCollector=collector,
                                                      logger=log)
        # save params
        ckpt_name = 'model_bert_squad_quantized_{0}'.format(calib_mode)
        params_saved = os.path.join(output_dir, ckpt_name)
        net.export(params_saved, epoch=0)
        log.info('Saving quantized model at %s', output_dir)
    
    def evaluate():
        """Evaluate the model on validation dataset."""
        log.info('start prediction')
        all_results = collections.defaultdict(list)
    
        epoch_tic = time.time()
        total_num = 0
        for data in dev_dataloader:
            example_ids, inputs, token_types, valid_length, _, _ = data
            total_num += len(inputs)
            out = net(inputs.as_in_context(ctx),
                      token_types.as_in_context(ctx),
                      valid_length.as_in_context(ctx).astype('float32'))
    
            output = mx.nd.split(out, axis=2, num_outputs=2)
            example_ids = example_ids.asnumpy().tolist()
            pred_start = output[0].reshape((0, -3)).asnumpy()
            pred_end = output[1].reshape((0, -3)).asnumpy()
    
            for example_id, start, end in zip(example_ids, pred_start, pred_end):
                all_results[example_id].append(PredResult(start=start, end=end))
    
        epoch_toc = time.time()
        log.info('Time cost={:.2f} s, Thoughput={:.2f} samples/s'.format(
            epoch_toc - epoch_tic, total_num / (epoch_toc - epoch_tic)))
    
        log.info('Get prediction results...')
    
        all_predictions = collections.OrderedDict()
    
        for features in dev_dataset:
            results = all_results[features[0].example_id]
            example_qas_id = features[0].qas_id
    
            prediction, _ = predict(
                features=features,
                results=results,
                tokenizer=nlp.data.BERTBasicTokenizer(lower=lower),
                max_answer_length=max_answer_length,
                null_score_diff_threshold=null_score_diff_threshold,
                n_best_size=n_best_size,
                version_2=version_2)
    
            all_predictions[example_qas_id] = prediction
    
        F1_EM = get_F1_EM(dev_dataset, all_predictions)
        log.info(F1_EM)
    
        with io.open(os.path.join(output_dir, 'predictions.json'),
                     'w', encoding='utf-8') as fout:
            data = json.dumps(all_predictions, ensure_ascii=False)
            fout.write(data)
     
        
        
    if only_calibration:
        try:
            calibration(net,
                        num_calib_batches,
                        quantized_dtype,
                        calib_mode)
        except AttributeError:
            nlp.utils.version.check_version('1.7.0', warning_only=True, library=mx)
            warnings.warn('INT8 Quantization for BERT need mxnet-mkl >= 1.6.0b20200115')
    elif not only_predict:
        evaluate()
