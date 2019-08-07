import mxnet as mx
import psutil
from mxnet.gluon import nn

from autogluon.estimator import *
from autogluon.estimator import Estimator
from autogluon.scheduler.reporter import StatusReporter
from .dataset import *
from .event_handlers import TextDataLoaderHandler
from .losses import get_loss_instance
from .metrics import get_metric_instance
from .model_zoo import get_model_instances, LMClassifier, BERTClassifier
from ...basic import autogluon_method

__all__ = ['train_text_classification']
logger = logging.getLogger(__name__)


def _get_bert_pre_trained_model(args: dict, ctx):
    """
    :param args:
    :param ctx:
    :return: net,vocab,
    """

    kwargs = {'use_pooler': True,
              'use_decoder': False,
              'use_classifier': False}

    pre_trained_network, vocab = get_model_instances(name=args.model, pretrained=args.pretrained, ctx=ctx, **kwargs)

    net = BERTClassifier(bert=pre_trained_network, num_classes=args.data.num_classes, dropout=args.dropout)

    return net, vocab


def _get_lm_pre_trained_model(args: dict, ctx):
    """
    Utility method which defines a Language Model for classification and also initializes
    dataset object compatible with Language Model.

    :param args:
    :param batch_size:
    :param ctx:
    :return: net, dataset, model_handlers
    """
    pre_trained_network, vocab = get_model_instances(name=args.model, pretrained=args.pretrained, ctx=ctx)

    net = LMClassifier()
    net.embedding = pre_trained_network.embedding
    net.encoder = pre_trained_network.encoder

    return net, vocab


@autogluon_method
def train_text_classification(args: dict, reporter: StatusReporter, task_id: int, resources=None) -> None:
    # Set Hyper-params
    def _init_hparams():
        batch_size = args.data.batch_size * max(args.num_gpus, 1)
        ctx = [mx.gpu(i)
               for i in range(args.num_gpus)] if args.num_gpus > 0 else [mx.cpu()]
        return batch_size, ctx

    batch_size, ctx = _init_hparams()
    vars(args).update({'task_id': task_id})
    logger.info('Task ID : {0}, args : {1}, resources:{2}, pid:{3}'.format(task_id, args, resources, os.getpid()))

    ps_p = psutil.Process(os.getpid())
    ps_p.cpu_affinity(resources.cpu_ids)

    if 'bert' in args.model:
        net, vocab = _get_bert_pre_trained_model(args, ctx)
    elif 'lstm_lm' in args.model:  # Get LM specific model attributes
        net, vocab = _get_lm_pre_trained_model(args, ctx)
        net.classifier = nn.Sequential()
        with net.classifier.name_scope():
            net.classifier.add(nn.Dropout(args.dropout))
            net.classifier.add(nn.Dense(args.data.num_classes))

    else:
        raise ValueError('Unsupported pre-trained model type. {}  will be supported in the future.'.format(args.model))

    if not args.pretrained:
        net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    else:
        net.classifier.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

    net.collect_params().reset_ctx(ctx)
    net.hybridize(static_alloc=True)

    # do not apply weight decay on LayerNorm and bias terms
    for _, v in net.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0

    def _get_dataloader():
        def _init_dataset(dataset, transform_fn):
            return transform(dataset, transform_fn, args.data.num_workers)

        class_labels = args.data.class_labels if args.data.class_labels else list(args.data._label_set)

        train_dataset = _init_dataset(args.data.train,
                                      get_transform_train_fn(args.model, vocab, args.max_sequence_length,
                                                             args.data.pair, class_labels))
        val_dataset = _init_dataset(args.data.val,
                                    get_transform_val_fn(args.model, vocab, args.max_sequence_length, args.data.pair,
                                                         class_labels))

        train_data = gluon.data.DataLoader(dataset=train_dataset, num_workers=args.data.num_workers,
                                           batch_sampler=get_batch_sampler(args.model, train_dataset, batch_size,
                                                                           args.data.num_workers),
                                           batchify_fn=get_batchify_fn(args.model))

        val_data = gluon.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
                                         batchify_fn=get_batchify_fn(args.model),
                                         num_workers=args.data.num_workers,
                                         shuffle=False)

        return train_data, val_data

    train_data, val_data = _get_dataloader()

    # fine_tune_lm(pre_trained_network) # TODO

    def _get_optimizer_params():
        # TODO : Add more optimizer params based on the chosen optimizer
        optimizer_params = {'learning_rate': args.lr}
        return optimizer_params

    optimer_params = _get_optimizer_params()

    trainer = gluon.Trainer(net.collect_params(), args.optimizer, optimer_params)

    # TODO : Update with search space
    loss = get_loss_instance(args.loss)
    metric = get_metric_instance(args.metric)

    estimator: Estimator = Estimator(net=net, loss=loss, metrics=[metric], trainer=trainer, context=ctx)

    early_stopping_handler = EarlyStoppingHandler(monitor=estimator.train_metrics[0], mode='max')

    lr_handler = LRHandler(warmup_ratio=0.1,
                           batch_size=batch_size,
                           num_epochs=args.epochs,
                           train_length=len(args.data.train))

    event_handlers = [early_stopping_handler, lr_handler, TextDataLoaderHandler(args.model), ReporterHandler(reporter)]

    estimator.fit(train_data=train_data, val_data=val_data, epochs=args.epochs,
                  event_handlers=event_handlers)
