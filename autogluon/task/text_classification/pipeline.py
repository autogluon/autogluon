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
import gluonnlp as nlp
from gluonnlp.data import BERTTokenizer
from gluonnlp.model import BERTClassifier, RoBERTaClassifier
# from autogluon.estimator import *
# from autogluon.estimator import Estimator
# from autogluon.scheduler.reporter import StatusReporter
from .dataset import *
# from .event_handlers import TextDataLoaderHandler
from .losses import get_loss_instance
from .metrics import get_metric_instance
from .model_zoo import get_model_instances, LMClassifier, BERTClassifier
from .transforms import BERTDatasetTransform
from ...basic import autogluon_method

__all__ = ['train_text_classification']
logger = logging.getLogger(__name__)

@autogluon_method
def train_text_classification(args, reporter=None):
    # Step 1: add scripts every function and python objects in the original training script except for the training function
    # at the beginning of the decorated function

    tasks = {
        'MRPC': MRPCTask(),
        'QQP': QQPTask(),
        'QNLI': QNLITask(),
        'RTE': RTETask(),
        'STS-B': STSBTask(),
        'CoLA': CoLATask(),
        'MNLI': MNLITask(),
        'WNLI': WNLITask(),
        'SST': SSTTask(),
        'IMDB': nlp.data.IMDB(), #TODO: metric in args not in the task
    }
    batch_size = args.batch_size
    dev_batch_size = args.dev_batch_size
    task_name = args.task_name
    lr = args.lr
    epsilon = args.epsilon
    accumulate = args.accumulate
    log_interval = args.log_interval * accumulate if accumulate else args.log_interval
    if accumulate:
        logging.info('Using gradient accumulation. Effective batch size = ' \
                     'batch_size * accumulate = %d', accumulate * batch_size)

    # random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    mx.random.seed(args.seed)

    ctx = mx.cpu() if args.gpu is None else mx.gpu(0) # TODO: simple fix

    task = tasks[task_name]

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
            logging.info('Mixed precision training with float16 requires MXNet >= '
                         '1.5.0b20190627. Please consider upgrading your MXNet version.')
            exit()

    # model and loss
    only_inference = args.only_inference
    model_name = args.bert_model
    dataset = args.bert_dataset
    pretrained_bert_parameters = args.pretrained_bert_parameters
    model_parameters = args.model_parameters
    if only_inference and not model_parameters:
        warnings.warn('model_parameters is not set. '
                      'Randomly initialized model will be used for inference.')

    get_pretrained = not (pretrained_bert_parameters is not None
                          or model_parameters is not None)

    use_roberta = 'roberta' in model_name
    get_model_params = {
        'name': model_name,
        'dataset_name': dataset,
        'pretrained': get_pretrained,
        'ctx': ctx,
        'use_decoder': False,
        'use_classifier': False,
    }
    # RoBERTa does not contain parameters for sentence pair classification
    if not use_roberta:
        get_model_params['use_pooler'] = True

    bert, vocabulary = nlp.model.get_model(**get_model_params)

    # initialize the rest of the parameters
    initializer = mx.init.Normal(0.02)
    # STS-B is a regression task.
    # STSBTask().class_labels returns None
    do_regression = not task.class_labels
    if do_regression:
        num_classes = 1
        loss_function = gluon.loss.L2Loss()
    else:
        num_classes = len(task.class_labels)
        loss_function = gluon.loss.SoftmaxCELoss()
    # reuse the BERTClassifier class with num_classes=1 for regression
    if use_roberta:
        model = RoBERTaClassifier(bert, dropout=0.0, num_classes=num_classes)
    else:
        model = BERTClassifier(bert, dropout=0.1, num_classes=num_classes)
    # initialize classifier
    if not model_parameters:
        model.classifier.initialize(init=initializer, ctx=ctx)

    # load checkpointing
    output_dir = args.output_dir
    if pretrained_bert_parameters:
        logging.info('loading bert params from %s', pretrained_bert_parameters)
        nlp.utils.load_parameters(model.bert, pretrained_bert_parameters, ctx=ctx,
                                  ignore_extra=True, cast_dtype=True)
    if model_parameters:
        logging.info('loading model params from %s', model_parameters)
        nlp.utils.load_parameters(model, model_parameters, ctx=ctx, cast_dtype=True)
    nlp.utils.mkdir(output_dir)

    logging.debug(model)
    model.hybridize(static_alloc=True)
    loss_function.hybridize(static_alloc=True)

    # data processing
    do_lower_case = 'uncased' in dataset
    if use_roberta:
        bert_tokenizer = nlp.data.GPT2BPETokenizer()
    else:
        bert_tokenizer = BERTTokenizer(vocabulary, lower=do_lower_case)

    def preprocess_data(tokenizer, task, batch_size, dev_batch_size, max_len, vocab, pad=False):
        """Train/eval Data preparation function."""
        pool = multiprocessing.Pool()

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
        pad_val = vocabulary[vocabulary.padding_token]
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
            num_workers=4,
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
                num_workers=4,
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
        test_tsv = task.dataset_test()
        test_tsv_list = test_tsv if isinstance(test_tsv, list) else [test_tsv]
        loader_test_list = []
        for segment, data in test_tsv_list:
            data_test = mx.gluon.data.SimpleDataset(pool.map(test_trans, data))
            loader_test = mx.gluon.data.DataLoader(
                data_test,
                batch_size=dev_batch_size,
                num_workers=4,
                shuffle=False,
                batchify_fn=test_batchify_fn)
            loader_test_list.append((segment, loader_test))
        return loader_train, loader_dev_list, loader_test_list, len(data_train)

    # Get the loader.
    logging.info('processing dataset...')
    train_data, dev_data_list, test_data_list, num_train_examples = preprocess_data(
        bert_tokenizer, task, batch_size, dev_batch_size, args.max_len, vocabulary, args.pad)

    def test(loader_test, segment):
        """Inference function on the test dataset."""
        logging.info('Now we are doing testing on %s with %s.', segment, ctx)

        tic = time.time()
        results = []
        for _, seqs in enumerate(loader_test):
            input_ids, valid_length, segment_ids = seqs
            input_ids = input_ids.as_in_context(ctx)
            valid_length = valid_length.as_in_context(ctx).astype('float32')
            if use_roberta:
                out = model(input_ids, valid_length)
            else:
                out = model(input_ids, segment_ids.as_in_context(ctx), valid_length)
            if not task.class_labels:
                # regression task
                for result in out.asnumpy().reshape(-1).tolist():
                    results.append('{:.3f}'.format(result))
            else:
                # classification task
                indices = mx.nd.topk(out, k=1, ret_typ='indices', dtype='int32').asnumpy()
                for index in indices:
                    results.append(task.class_labels[int(index)])

        mx.nd.waitall()
        toc = time.time()
        logging.info('Time cost=%.2fs, throughput=%.2f samples/s', toc - tic,
                     dev_batch_size * len(loader_test) / (toc - tic))
        # write result to a file.
        segment = segment.replace('_mismatched', '-mm')
        segment = segment.replace('_matched', '-m')
        segment = segment.replace('SST', 'SST-2')
        filename = args.task_name + segment.replace('test', '') + '.tsv'
        test_path = os.path.join(args.output_dir, filename)
        with io.open(test_path, 'w', encoding='utf-8') as f:
            f.write(u'index\tprediction\n')
            for i, pred in enumerate(results):
                f.write(u'%d\t%s\n' % (i, str(pred)))

    def log_train(batch_id, batch_num, metric, step_loss, log_interval, epoch_id, learning_rate):
        """Generate and print out the log message for training. """
        metric_nm, metric_val = metric.get()
        if not isinstance(metric_nm, list):
            metric_nm, metric_val = [metric_nm], [metric_val]

        train_str = '[Epoch %d Batch %d/%d] loss=%.4f, lr=%.7f, metrics:' + \
                    ','.join([i + ':%.4f' for i in metric_nm])
        logging.info(train_str, epoch_id + 1, batch_id + 1, batch_num,
                     step_loss / log_interval, learning_rate, *metric_val)

    def log_eval(batch_id, batch_num, metric, step_loss, log_interval):
        """Generate and print out the log message for inference. """
        metric_nm, metric_val = metric.get()
        if not isinstance(metric_nm, list):
            metric_nm, metric_val = [metric_nm], [metric_val]

        eval_str = '[Batch %d/%d] loss=%.4f, metrics:' + \
                   ','.join([i + ':%.4f' for i in metric_nm])
        logging.info(eval_str, batch_id + 1, batch_num,
                     step_loss / log_interval, *metric_val)

    def evaluate(loader_dev, metric, segment):
        """Evaluate the model on validation dataset."""
        logging.info('Now we are doing evaluation on %s with %s.', segment, ctx)
        metric.reset()
        step_loss = 0
        tic = time.time()
        for batch_id, seqs in enumerate(loader_dev):
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
                log_eval(batch_id, len(loader_dev), metric, step_loss, args.log_interval)
                step_loss = 0

        metric_nm, metric_val = metric.get()
        if not isinstance(metric_nm, list):
            metric_nm, metric_val = [metric_nm], [metric_val]
        metric_str = 'validation metrics:' + ','.join([i + ':%.4f' for i in metric_nm])
        logging.info(metric_str, *metric_val)

        mx.nd.waitall()
        toc = time.time()
        logging.info('Time cost=%.2fs, throughput=%.2f samples/s', toc - tic,
                     dev_batch_size * len(loader_dev) / (toc - tic))
        return metric_nm, metric_val

    # Step 2: the training function in the original training script is added in the decorated function in autogluon for training.

    """Training function."""
    if not only_inference:
        logging.info('Now we are doing BERT classification training on %s!', ctx)

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
            logging.info('Early stopping at epoch %d', epoch_id)
            break
        if not only_inference:
            task.metric.reset()
            step_loss = 0
            tic = time.time()
            all_model_params.zero_grad()

            for batch_id, seqs in enumerate(train_data):
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
                              epoch_id, trainer.learning_rate)
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
                reporter(epoch=epoch_id, accuracy=metric_val[0])

        if not only_inference:
            # save params
            ckpt_name = 'model_bert_{0}_{1}.params'.format(task_name, epoch_id)
            params_saved = os.path.join(output_dir, ckpt_name)

            nlp.utils.save_parameters(model, params_saved)
            logging.info('params saved in: %s', params_saved)
            toc = time.time()
            logging.info('Time cost=%.2fs', toc - tic)
            tic = toc

    if not only_inference:
        # we choose the best model based on metric[0],
        # assuming higher score stands for better model quality
        metric_history.sort(key=lambda x: x[2][0], reverse=True)
        epoch_id, metric_nm, metric_val = metric_history[0]
        ckpt_name = 'model_bert_{0}_{1}.params'.format(task_name, epoch_id)
        params_saved = os.path.join(output_dir, ckpt_name)
        nlp.utils.load_parameters(model, params_saved)
        metric_str = 'Best model at epoch {}. Validation metrics:'.format(epoch_id)
        metric_str += ','.join([i + ':%.4f' for i in metric_nm])
        logging.info(metric_str, *metric_val)

    # inference on test data
    for segment, test_data in test_data_list:
        test(test_data, segment)

@autogluon_method
def train_text_classification(args, reporter=None):
    """Training function."""
    model_name = args.bert_model
    dataset_name = args.bert_dataset
    only_predict = args.only_predict
    model_parameters = args.model_parameters
    pretrained_bert_parameters = args.pretrained_bert_parameters
    if pretrained_bert_parameters and model_parameters:
        raise ValueError('Cannot provide both pre-trained BERT parameters and '
                         'BertForQA model parameters.')
    lower = args.uncased

    epochs = args.epochs
    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    lr = args.lr
    ctx = mx.cpu() if args.gpu is None else mx.gpu(args.gpu)

    accumulate = args.accumulate
    log_interval = args.log_interval * accumulate if accumulate else args.log_interval
    if accumulate:
        log.info('Using gradient accumulation. Effective batch size = {}'.
                 format(accumulate * batch_size))

    optimizer = args.optimizer
    warmup_ratio = args.warmup_ratio

    version_2 = args.version_2
    null_score_diff_threshold = args.null_score_diff_threshold

    max_seq_length = args.max_seq_length
    doc_stride = args.doc_stride
    max_query_length = args.max_query_length
    n_best_size = args.n_best_size
    max_answer_length = args.max_answer_length

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

    pretrained = not model_parameters and not pretrained_bert_parameters and not args.sentencepiece
    bert, vocab = nlp.model.get_model(
        name=model_name,
        dataset_name=dataset_name,
        vocab=vocab,
        pretrained=pretrained,
        ctx=ctx,
        use_pooler=False,
        use_decoder=False,
        use_classifier=False)

    if args.sentencepiece:
        tokenizer = nlp.data.BERTSPTokenizer(args.sentencepiece, vocab, lower=lower)
    else:
        tokenizer = nlp.data.BERTTokenizer(vocab=vocab, lower=lower)

    batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Stack(),
        nlp.data.batchify.Pad(axis=0, pad_val=vocab[vocab.padding_token]),
        nlp.data.batchify.Pad(axis=0, pad_val=vocab[vocab.padding_token]),
        nlp.data.batchify.Stack('float32'),
        nlp.data.batchify.Stack('float32'),
        nlp.data.batchify.Stack('float32'))

    net = BertForQA(bert=bert)
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

    net.hybridize(static_alloc=True)

    loss_function = BertForQALoss()
    loss_function.hybridize(static_alloc=True)

    segment = 'train' if not args.debug else 'dev'
    log.info('Loading %s data...', segment)
    if version_2:
        train_data = SQuAD(segment, version='2.0')
    else:
        train_data = SQuAD(segment, version='1.1')
    if args.debug:
        sampled_data = [train_data[i] for i in range(1000)]
        train_data = mx.gluon.data.SimpleDataset(sampled_data)
    log.info('Number of records in Train data:{}'.format(len(train_data)))

    train_data_transform, _ = preprocess_dataset(
        train_data, SQuADTransform(
            copy.copy(tokenizer),
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_pad=True,
            is_training=True))
    log.info('The number of examples after preprocessing:{}'.format(
        len(train_data_transform)))

    train_dataloader = mx.gluon.data.DataLoader(
        train_data_transform, batchify_fn=batchify_fn,
        batch_size=batch_size, num_workers=4, shuffle=True)

    log.info('Start Training')

    optimizer_params = {'learning_rate': lr}
    try:
        trainer = mx.gluon.Trainer(net.collect_params(), optimizer,
                                   optimizer_params, update_on_kvstore=False)
    except ValueError as e:
        print(e)
        warnings.warn('AdamW optimizer is not found. Please consider upgrading to '
                      'mxnet>=1.5.0. Now the original Adam optimizer is used instead.')
        trainer = mx.gluon.Trainer(net.collect_params(), 'adam',
                                   optimizer_params, update_on_kvstore=False)

    num_train_examples = len(train_data_transform)
    step_size = batch_size * accumulate if accumulate else batch_size
    num_train_steps = int(num_train_examples / step_size * epochs)
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    step_num = 0

    def set_new_lr(step_num, batch_id):
        """set new learning rate"""
        # set grad to zero for gradient accumulation
        if accumulate:
            if batch_id % accumulate == 0:
                net.collect_params().zero_grad()
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
    params = [p for p in net.collect_params().values()
              if p.grad_req != 'null']
    # Set grad_req if gradient accumulation is required
    if accumulate:
        for p in params:
            p.grad_req = 'add'

    epoch_tic = time.time()
    total_num = 0
    log_num = 0
    for epoch_id in range(epochs):
        step_loss = 0.0
        tic = time.time()
        for batch_id, data in enumerate(train_dataloader):
            # set new lr
            step_num = set_new_lr(step_num, batch_id)
            # forward and backward
            with mx.autograd.record():
                _, inputs, token_types, valid_length, start_label, end_label = data

                log_num += len(inputs)
                total_num += len(inputs)

                out = net(inputs.astype('float32').as_in_context(ctx),
                          token_types.astype('float32').as_in_context(ctx),
                          valid_length.astype('float32').as_in_context(ctx))

                ls = loss_function(out, [
                    start_label.astype('float32').as_in_context(ctx),
                    end_label.astype('float32').as_in_context(ctx)]).mean()

                if accumulate:
                    ls = ls / accumulate
            ls.backward()
            # update
            if not accumulate or (batch_id + 1) % accumulate == 0:
                trainer.allreduce_grads()
                nlp.utils.clip_grad_global_norm(params, 1)
                trainer.update(1)

            step_loss += ls.asscalar()

            if (batch_id + 1) % log_interval == 0:
                toc = time.time()
                log.info('Epoch: {}, Batch: {}/{}, Loss={:.4f}, lr={:.7f} Time cost={:.1f} Thoughput={:.2f} samples/s'  # pylint: disable=line-too-long
                         .format(epoch_id, batch_id, len(train_dataloader),
                                 step_loss / log_interval,
                                 trainer.learning_rate, toc - tic, log_num/(toc - tic)))
                tic = time.time()
                step_loss = 0.0
                log_num = 0
        epoch_toc = time.time()
        log.info('Time cost={:.2f} s, Thoughput={:.2f} samples/s'.format(
            epoch_toc - epoch_tic, total_num/(epoch_toc - epoch_tic)))

    net.save_parameters(os.path.join(output_dir, 'net.params'))


def evaluate():
    """Evaluate the model on validation dataset.
    """
    log.info('Loading dev data...')
    if version_2:
        dev_data = SQuAD('dev', version='2.0')
    else:
        dev_data = SQuAD('dev', version='1.1')
    if args.debug:
        sampled_data = [dev_data[0], dev_data[1], dev_data[2]]
        dev_data = mx.gluon.data.SimpleDataset(sampled_data)
    log.info('Number of records in dev data:{}'.format(len(dev_data)))

    dev_dataset = dev_data.transform(
        SQuADTransform(
            copy.copy(tokenizer),
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_pad=False,
            is_training=False)._transform, lazy=False)

    dev_data_transform, _ = preprocess_dataset(
        dev_data, SQuADTransform(
            copy.copy(tokenizer),
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_pad=False,
            is_training=False))
    log.info('The number of examples after preprocessing:{}'.format(
        len(dev_data_transform)))

    dev_dataloader = mx.gluon.data.DataLoader(
        dev_data_transform,
        batchify_fn=batchify_fn,
        num_workers=4, batch_size=test_batch_size,
        shuffle=False, last_batch='keep')

    log.info('start prediction')

    all_results = collections.defaultdict(list)

    epoch_tic = time.time()
    total_num = 0
    for data in dev_dataloader:
        example_ids, inputs, token_types, valid_length, _, _ = data
        total_num += len(inputs)
        out = net(inputs.astype('float32').as_in_context(ctx),
                  token_types.astype('float32').as_in_context(ctx),
                  valid_length.astype('float32').as_in_context(ctx))

        output = mx.nd.split(out, axis=2, num_outputs=2)
        example_ids = example_ids.asnumpy().tolist()
        pred_start = output[0].reshape((0, -3)).asnumpy()
        pred_end = output[1].reshape((0, -3)).asnumpy()

        for example_id, start, end in zip(example_ids, pred_start, pred_end):
            all_results[example_id].append(PredResult(start=start, end=end))

    epoch_toc = time.time()
    log.info('Time cost={:.2f} s, Thoughput={:.2f} samples/s'.format(
        epoch_toc - epoch_tic, total_num/(epoch_toc - epoch_tic)))

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

    with io.open(os.path.join(output_dir, 'predictions.json'),
                 'w', encoding='utf-8') as fout:
        data = json.dumps(all_predictions, ensure_ascii=False)
        fout.write(data)

    if version_2:
        log.info('Please run evaluate-v2.0.py to get evaluation results for SQuAD 2.0')
    else:
        F1_EM = get_F1_EM(dev_data, all_predictions)
        log.info(F1_EM)
