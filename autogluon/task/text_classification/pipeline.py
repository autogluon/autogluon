import os
import io
import time
import logging
import multiprocessing
import random
import warnings
import numpy as np

import mxnet as mx
from mxnet import gluon, init, autograd, nd
from mxnet.gluon import nn
from ...utils.try_import import try_import_gluonnlp


from .network import get_network
from .dataset import *
from .transforms import BERTDatasetTransform
from ...core import *
from ...utils import tqdm
from ...utils.mxutils import collect_params

__all__ = ['train_text_classification', 'preprocess_data']


def preprocess_data(tokenizer, task, batch_size, dev_batch_size, max_len, vocab, pad=False, num_workers=1):
    """Train/eval Data preparation function."""
    pool = multiprocessing.Pool()
    nlp = try_import_gluonnlp()
    # transformation for data train and dev
    label_dtype = 'float32' if not task.class_labels else 'int32'
    trans = BERTDatasetTransform(tokenizer, max_len,
                                 vocab=vocab,
                                 class_labels=task.class_labels,
                                 label_alias=task.label_alias,
                                 pad=pad, pair=task.is_pair,
                                 has_label=True)

    # data train
    # task.dataset_train returns (segment_name, dataset)
    train_tsv = task.dataset_train()[1]
    data_train = mx.gluon.data.SimpleDataset(pool.map(trans, train_tsv))
    data_train_len = data_train.transform(
        lambda input_id, length, segment_id, label_id: length, lazy=False)
    # bucket sampler for training
    pad_val = vocab[vocab.padding_token]
    batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0, pad_val=pad_val),  # input
        nlp.data.batchify.Stack(),  # length
        nlp.data.batchify.Pad(axis=0, pad_val=0),  # segment
        nlp.data.batchify.Stack(label_dtype))  # label
    batch_sampler = nlp.data.sampler.FixedBucketSampler(
        data_train_len,
        batch_size=batch_size,
        num_buckets=10,
        ratio=0,
        shuffle=True)
    # data loader for training
    loader_train = gluon.data.DataLoader(
        dataset=data_train,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        batchify_fn=batchify_fn)

    # data dev. For MNLI, more than one dev set is available
    dev_tsv = task.dataset_dev()
    dev_tsv_list = dev_tsv if isinstance(dev_tsv, list) else [dev_tsv]
    loader_dev_list = []
    for segment, data in dev_tsv_list:
        data_dev = mx.gluon.data.SimpleDataset(pool.map(trans, data))
        loader_dev = mx.gluon.data.DataLoader(
            data_dev,
            batch_size=dev_batch_size,
            num_workers=num_workers,
            shuffle=False,
            batchify_fn=batchify_fn)
        loader_dev_list.append((segment, loader_dev))

    # batchify for data test
    test_batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0, pad_val=pad_val), nlp.data.batchify.Stack(),
        nlp.data.batchify.Pad(axis=0, pad_val=0))
    # transform for data test
    test_trans = BERTDatasetTransform(tokenizer, max_len,
                                      vocab=vocab,
                                      class_labels=None,
                                      pad=pad, pair=task.is_pair,
                                      has_label=False)

    # data test. For MNLI, more than one test set is available
    #test_tsv = task.dataset_test()
    #test_tsv_list = test_tsv if isinstance(test_tsv, list) else [test_tsv]
    #loader_test_list = []
    #for segment, data in test_tsv_list:
    #    data_test = mx.gluon.data.SimpleDataset(pool.map(test_trans, data))
    #    loader_test = mx.gluon.data.DataLoader(
    #        data_test,
    #        batch_size=dev_batch_size,
    #        num_workers=num_workers,
    #        shuffle=False,
    #        batchify_fn=test_batchify_fn)
    #    loader_test_list.append((segment, loader_test))
    return loader_train, loader_dev_list, len(data_train), trans, test_trans

@args()
def train_text_classification(args, reporter=None):
    # Step 1: add scripts every function and python objects in the original training script except for the training function
    # at the beginning of the decorated function
    nlp = try_import_gluonnlp()
    logger = logging.getLogger(__name__)
    if args.verbose:
        logger.setLevel(logging.INFO)
        logger.info(args)
    batch_size = args.batch_size
    dev_batch_size = args.dev_batch_size
    lr = args.lr
    epsilon = args.epsilon
    accumulate = args.accumulate
    log_interval = args.log_interval * accumulate if accumulate else args.log_interval
    if accumulate:
        logger.info('Using gradient accumulation. Effective batch size = ' \
                     'batch_size * accumulate = %d', accumulate * batch_size)

    # random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    mx.random.seed(args.seed)

    # TODO support for multi-GPU
    ctx = [mx.gpu(i) for i in range(args.num_gpus)][0] if args.num_gpus > 0 else [mx.cpu()][0]

    task = args.dataset
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

    # model and loss
    model_name = args.net
    dataset = args.pretrained_dataset

    use_roberta = 'roberta' in model_name
    get_model_params = {
        'name': model_name,
        'dataset_name': dataset,
        'pretrained': True,
        'ctx': ctx,
        'use_decoder': False,
        'use_classifier': False,
    }
    # RoBERTa does not contain parameters for sentence pair classification
    if not use_roberta:
        get_model_params['use_pooler'] = True

    bert, vocabulary = nlp.model.get_model(**get_model_params)
    model = get_network(bert, task.class_labels, use_roberta)
    #do_regression = not task.class_labels
    #if do_regression:
    #    num_classes = 1
    #    loss_function = gluon.loss.L2Loss()
    #else:
    #    num_classes = len(task.class_labels)
    #    loss_function = gluon.loss.SoftmaxCELoss()
    ## reuse the BERTClassifier class with num_classes=1 for regression
    #if use_roberta:
    #    model = RoBERTaClassifier(bert, dropout=0.0, num_classes=num_classes)
    #else:
    #    model = BERTClassifier(bert, dropout=0.1, num_classes=num_classes)
    # initialize classifier
    loss_function = gluon.loss.SoftmaxCELoss() if task.class_labels else gluon.loss.L2Loss()
    initializer = mx.init.Normal(0.02)
    model.classifier.initialize(init=initializer, ctx=ctx)

    model.hybridize(static_alloc=True)
    loss_function.hybridize(static_alloc=True)

    # data processing
    do_lower_case = 'uncased' in dataset
    if use_roberta:
        bert_tokenizer = nlp.data.GPT2BPETokenizer()
    else:
        bert_tokenizer = nlp.data.BERTTokenizer(vocabulary, lower=do_lower_case)


    # Get the loader.
    train_data, dev_data_list, num_train_examples, trans, test_trans = preprocess_data(
        bert_tokenizer, task, batch_size, dev_batch_size, args.max_len, vocabulary,
        True, args.num_workers)

    def log_train(batch_id, batch_num, metric, step_loss, log_interval, epoch_id, learning_rate, tbar):
        """Generate and print out the log message for training. """
        metric_nm, metric_val = metric.get()
        if not isinstance(metric_nm, list):
            metric_nm, metric_val = [metric_nm], [metric_val]

        train_str = '[Epoch %d] loss=%.4f, lr=%.7f, metrics:' + \
                    ','.join([i + ':%.4f' for i in metric_nm])
        tbar.set_description(train_str % (epoch_id, step_loss / log_interval, learning_rate, *metric_val))

    def log_eval(batch_id, batch_num, metric, step_loss, log_interval, tbar):
        """Generate and print out the log message for inference. """
        metric_nm, metric_val = metric.get()
        if not isinstance(metric_nm, list):
            metric_nm, metric_val = [metric_nm], [metric_val]

        eval_str = 'loss=%.4f, metrics:' + \
                   ','.join([i + ':%.4f' for i in metric_nm])
        tbar.set_description(eval_str % (step_loss / log_interval, *metric_val))

    def evaluate(loader_dev, metric, segment):
        """Evaluate the model on validation dataset."""
        metric.reset()
        step_loss = 0
        tbar = tqdm(loader_dev)
        for batch_id, seqs in enumerate(tbar):
            input_ids, valid_length, segment_ids, label = seqs
            input_ids = input_ids.as_in_context(ctx)
            valid_length = valid_length.as_in_context(ctx).astype('float32')
            label = label.as_in_context(ctx)
            if use_roberta:
                out = model(input_ids, valid_length)
            else:
                out = model(input_ids, segment_ids.as_in_context(ctx), valid_length)
            ls = loss_function(out, label).mean()

            step_loss += ls.asscalar()
            metric.update([label], [out])

            if (batch_id + 1) % (args.log_interval) == 0:
                log_eval(batch_id, len(loader_dev), metric, step_loss, args.log_interval, tbar)
                step_loss = 0

        metric_nm, metric_val = metric.get()
        if not isinstance(metric_nm, list):
            metric_nm, metric_val = [metric_nm], [metric_val]
        metric_str = 'validation metrics:' + ','.join([i + ':%.4f' for i in metric_nm])
        logger.info(metric_str, *metric_val)

        mx.nd.waitall()
        return metric_nm, metric_val

    # Step 2: the training function in the original training script is added in the decorated function in autogluon for training.
    """Training function."""

    all_model_params = model.collect_params()
    optimizer_params = {'learning_rate': lr, 'epsilon': epsilon, 'wd': 0.01}
    trainer = gluon.Trainer(all_model_params, 'bertadam',
                            optimizer_params, update_on_kvstore=False)
    if args.dtype == 'float16':
        amp.init_trainer(trainer)

    step_size = batch_size * accumulate if accumulate else batch_size
    num_train_steps = int(num_train_examples / step_size * args.epochs)
    warmup_ratio = args.warmup_ratio
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    step_num = 0

    # Do not apply weight decay on LayerNorm and bias terms
    for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    # Collect differentiable parameters
    params = [p for p in all_model_params.values() if p.grad_req != 'null']

    # Set grad_req if gradient accumulation is required
    if accumulate and accumulate > 1:
        for p in params:
            p.grad_req = 'add'
    # track best eval score
    metric_history = []
    best_metric = None
    patience = args.early_stop

    tic = time.time()
    for epoch_id in range(args.epochs):
        if args.early_stop and patience == 0:
            logger.info('Early stopping at epoch %d', epoch_id)
            break
        task.metric.reset()
        step_loss = 0
        tic = time.time()
        all_model_params.zero_grad()

        tbar = tqdm(train_data)
        for batch_id, seqs in enumerate(tbar):
            # learning rate schedule
            if step_num < num_warmup_steps:
                new_lr = lr * step_num / num_warmup_steps
            else:
                non_warmup_steps = step_num - num_warmup_steps
                offset = non_warmup_steps / (num_train_steps - num_warmup_steps)
                new_lr = lr - offset * lr
            trainer.set_learning_rate(new_lr)

            # forward and backward
            with mx.autograd.record():
                input_ids, valid_length, segment_ids, label = seqs
                input_ids = input_ids.as_in_context(ctx)
                valid_length = valid_length.as_in_context(ctx).astype('float32')
                label = label.as_in_context(ctx)
                if use_roberta:
                    out = model(input_ids, valid_length)
                else:
                    out = model(input_ids, segment_ids.as_in_context(ctx), valid_length)
                ls = loss_function(out, label).mean()
                if args.dtype == 'float16':
                    with amp.scale_loss(ls, trainer) as scaled_loss:
                        mx.autograd.backward(scaled_loss)
                else:
                    ls.backward()

            # update
            if not accumulate or (batch_id + 1) % accumulate == 0:
                trainer.allreduce_grads()
                nlp.utils.clip_grad_global_norm(params, 1)
                trainer.update(accumulate if accumulate else 1)
                step_num += 1
                if accumulate and accumulate > 1:
                    # set grad to zero for gradient accumulation
                    all_model_params.zero_grad()

            step_loss += ls.asscalar()
            task.metric.update([label], [out])
            if (batch_id + 1) % (args.log_interval) == 0:
                log_train(batch_id, len(train_data), task.metric, step_loss, args.log_interval,
                          epoch_id, trainer.learning_rate, tbar)
                step_loss = 0
        mx.nd.waitall()

        # inference on dev data
        for segment, dev_data in dev_data_list:
            metric_nm, metric_val = evaluate(dev_data, task.metric, segment)
            if best_metric is None or metric_val >= best_metric:
                best_metric = metric_val
                patience = args.early_stop
            else:
                if args.early_stop is not None:
                    patience -= 1
            metric_history.append((epoch_id, metric_nm, metric_val))

        if reporter is not None:
            # Note: epoch reported back must start with 1, not with 0
            reporter(epoch=epoch_id+1, accuracy=metric_val[0])

    if args.final_fit:
        get_model_params.pop('ctx')
        return {'model_params': collect_params(model),
                'get_model_args': get_model_params,
                'class_labels': task.class_labels,
                'transform': trans,
                'test_transform': test_trans}
