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


# def _get_bert_pre_trained_model(args, ctx):
#     """
#     :param args:
#     :param ctx:
#     :return: net,vocab,
#     """
#
#     kwargs = {'use_pooler': True,
#               'use_decoder': False,
#               'use_classifier': False}
#
#     pre_trained_network, vocab = get_model_instances(name=args.model, pretrained=args.pretrained, ctx=ctx, **kwargs)
#
#     net = BERTClassifier(bert=pre_trained_network, num_classes=args.data.num_classes, dropout=args.dropout)
#     do_lower_case = 'uncased' in args.bert_dataset
#     bert_tokenizer = nlp.data.BERTTokenizer(vocab, lower=do_lower_case)
#
#     return net, vocab, bert_tokenizer
#
#
# def _get_lm_pre_trained_model(args: dict, ctx):
#     """
#     Utility method which defines a Language Model for classification and also initializes
#     dataset object compatible with Language Model.
#
#     :param args:
#     :param batch_size:
#     :param ctx:
#     :return: net, dataset, model_handlers
#     """
#     pre_trained_network, vocab = get_model_instances(name=args.model, pretrained=args.pretrained, ctx=ctx)
#
#     net = LMClassifier()
#     net.embedding = pre_trained_network.embedding
#     net.encoder = pre_trained_network.encoder
#
#     return net, vocab, _
#
# lr_schedulers = {
#     'poly': mx.lr_scheduler.PolyScheduler,
#     'cosine': mx.lr_scheduler.CosineScheduler
# }


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
            	reporter(epoch=epoch_id, accuracy=metric_val)

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

# @autogluon_method
# def train_text_classification(args, reporter, task_id, resources=None):
#     # Set Hyper-params
#     def _init_hparams():
#         batch_size = args.data.batch_size * max(args.num_gpus, 1)
#         ctx = [mx.gpu(i)
#                for i in range(args.num_gpus)] if args.num_gpus > 0 else [mx.cpu()]
#         return batch_size, ctx
#
#     batch_size, ctx = _init_hparams()
#     vars(args).update({'task_id': task_id})
#     logger.info('Task ID : {0}, args : {1}, resources:{2}, pid:{3}'.format(task_id, args, resources, os.getpid()))
#
#     ps_p = psutil.Process(os.getpid())
#     ps_p.cpu_affinity(resources.cpu_ids)
#
#     if 'bert' in args.model:
#         net, vocab, bert_tokenizer = _get_bert_pre_trained_model(args, ctx)
#     elif 'lstm_lm' in args.model:  # Get LM specific model attributes
#         net, vocab, _ = _get_lm_pre_trained_model(args, ctx)
#         net.classifier = nn.Sequential()
#         with net.classifier.name_scope():
#             net.classifier.add(nn.Dropout(args.dropout))
#             net.classifier.add(nn.Dense(args.data.num_classes))
#
#     else:
#         raise ValueError('Unsupported pre-trained model type. {}  will be supported in the future.'.format(args.model))
#
#     if not args.pretrained:
#         net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
#     else:
#         net.classifier.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
#
#     net.collect_params().reset_ctx(ctx)
#     net.hybridize(static_alloc=True)
#
#     # do not apply weight decay on LayerNorm and bias terms
#     for _, v in net.collect_params('.*beta|.*gamma|.*bias').items():
#         v.wd_mult = 0.0
#
#     # def _get_dataloader():
#     #     def _init_dataset(dataset, transform_fn):
#     #         return transform(dataset, transform_fn, args.data.num_workers)
#     #
#     #     class_labels = args.data.class_labels if args.data.class_labels else list(args.data._label_set)
#     #
#     #     train_dataset = _init_dataset(args.data.train,
#     #                                   get_transform_train_fn(args.model, vocab, args.max_sequence_length,
#     #                                                          args.data.pair, class_labels))
#     #     val_dataset = _init_dataset(args.data.val,
#     #                                 get_transform_val_fn(args.model, vocab, args.max_sequence_length, args.data.pair,
#     #                                                      class_labels))
#     #
#     #     train_data = gluon.data.DataLoader(dataset=train_dataset, num_workers=args.data.num_workers,
#     #                                        batch_sampler=get_batch_sampler(args.model, train_dataset, batch_size,
#     #                                                                        args.data.num_workers),
#     #                                        batchify_fn=get_batchify_fn(args.model))
#     #
#     #     val_data = gluon.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
#     #                                      batchify_fn=get_batchify_fn(args.model),
#     #                                      num_workers=args.data.num_workers,
#     #                                      shuffle=False)
#     #
#     #     return train_data, val_data
#     #
#     # train_data, val_data = _get_dataloader()
#
#     # fine_tune_lm(pre_trained_network) # TODO
#
#     # def _get_optimizer_params():
#     #     # TODO : Add more optimizer params based on the chosen optimizer
#     #     optimizer_params = {'learning_rate': args.lr}
#     #     return optimizer_params
#     #
#     # optimer_params = _get_optimizer_params()
#     #
#     # trainer = gluon.Trainer(net.collect_params(), args.optimizer, optimer_params)
#     task = tasks[args.task_name]
#     train_data, dev_data_list, test_data_list, num_train_examples = preprocess_data(
#         bert_tokenizer, task, batch_size, dev_batch_size, args.max_len, vocab, args.pad)
#
#     # Define trainer
#     def _set_optimizer_params(args):
#         # TODO (cgraywang): a better way?
#         if args.optimizer == 'sgd' or args.optimizer == 'nag':
#             if 'lr_scheduler' not in vars(args):
#                 optimizer_params = {
#                     'learning_rate': args.lr,
#                     'momentum': args.momentum,
#                     'wd': args.wd
#                 }
#             else:
#                 optimizer_params = {
#                     'lr_scheduler': lr_schedulers[args.lr_scheduler](len(train_data), base_lr=args.lr),
#                     'momentum': args.momentum,
#                     'wd': args.wd
#                 }
#         elif args.optimizer == 'adam':
#             if 'lr_scheduler' not in vars(args):
#                 optimizer_params = {
#                     'learning_rate': args.lr,
#                     'wd': args.wd
#                 }
#             else:
#                 optimizer_params = {
#                     'lr_scheduler': lr_schedulers[args.lr_scheduler](len(train_data), base_lr=args.lr),
#                     'wd': args.wd
#                 }
#         else:
#             raise NotImplementedError
#         return optimizer_params
#
#     optimizer_params = _set_optimizer_params(args)
#     trainer = gluon.Trainer(net.collect_params(),
#                             args.optimizer,
#                             optimizer_params)
#
#     # TODO : Update with search space
#     L = get_loss_instance(args.loss)
#     metric = get_metric_instance(args.metric)
#
#     # estimator: Estimator = Estimator(net=net, loss=loss, metrics=[metric], trainer=trainer, context=ctx)
#     #
#     # early_stopping_handler = EarlyStoppingHandler(monitor=estimator.train_metrics[0], mode='max')
#     #
#     # lr_handler = LRHandler(warmup_ratio=0.1,
#     #                        batch_size=batch_size,
#     #                        num_epochs=args.epochs,
#     #                        train_length=len(args.data.train))
#     #
#     # event_handlers = [early_stopping_handler, lr_handler, TextDataLoaderHandler(args.model), ReporterHandler(reporter)]
#     #
#     # estimator.fit(train_data=train_data, val_data=val_data, epochs=args.epochs,
#     #               event_handlers=event_handlers)
#
#     def _demo_early_stopping(batch_id):
#         if 'demo' in vars(args):
#             if args.demo and batch_id == 3:
#                 return True
#         return False
#
#     def train(epoch):
#         #TODO (cgraywang): change to lr scheduler
#         if hasattr(args, 'lr_step') and hasattr(args, 'lr_factor'):
#             if epoch % args.lr_step == 0:
#                 trainer.set_learning_rate(trainer.learning_rate * args.lr_factor)
#
#         for i, batch in enumerate(train_data):
#             data = gluon.utils.split_and_load(batch[0],
#                                               ctx_list=ctx,
#                                               batch_axis=0,
#                                               even_split=False)
#             label = gluon.utils.split_and_load(batch[1],
#                                                ctx_list=ctx,
#                                                batch_axis=0,
#                                                even_split=False)
#             with autograd.record():
#                 outputs = [net(X) for X in data]
#                 loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
#             for l in loss:
#                 l.backward()
#
#             trainer.step(batch_size)
#             if _demo_early_stopping(i):
#                 break
#         mx.nd.waitall()
#         #TODO: fix mutli gpu bug
#         # if reporter is not None:
#         #     reporter.save_dict(epoch=epoch, params=net.collect_params())
#
#     def test(epoch):
#         test_loss = 0
#         for i, batch in enumerate(val_data):
#             data = gluon.utils.split_and_load(batch[0],
#                                               ctx_list=ctx,
#                                               batch_axis=0,
#                                               even_split=False)
#             label = gluon.utils.split_and_load(batch[1],
#                                                ctx_list=ctx,
#                                                batch_axis=0,
#                                                even_split=False)
#             outputs = [net(X) for X in data]
#             loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
#
#             test_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
#             metric.update(label, outputs)
#             if _demo_early_stopping(i):
#                 break
#         _, test_acc = metric.get()
#         test_loss /= len(val_data)
#         # TODO (cgraywang): add ray
#         reporter(epoch=epoch, accuracy=test_acc, loss=test_loss)
#
#     for epoch in range(1, args.epochs + 1):
#         train(epoch)
#         if reporter is not None:
#             test(epoch)
#     if reporter is None:
#         net_path = os.path.join(os.path.splitext(args.savedir)[0], 'net.params')
#         net.save_parameters(net_path)
