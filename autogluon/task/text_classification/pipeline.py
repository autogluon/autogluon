import mxnet as mx
import psutil
from mxnet.gluon import nn

from autogluon.estimator import *
from autogluon.estimator import Estimator
from autogluon.scheduler.reporter import StatusReporter
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

    net = BERTClassifier()
    net.pre_trained_network = pre_trained_network

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
        """
        Method required to initialize context and batch size based on supplied arguments.
        :return:
        """
        if hasattr(args, 'batch_size') and hasattr(args, 'num_gpus'):
            batch_size = args.batch_size * max(args.num_gpus, 1)
            ctx = [mx.gpu(i)
                   for i in range(args.num_gpus)] if args.num_gpus > 0 else [mx.cpu()]
        else:
            if hasattr(args, 'num_gpus'):
                num_gpus = args.num_gpus
            else:
                num_gpus = 0
            if hasattr(args, 'batch_size'):
                batch_size = args.batch_size * max(num_gpus, 1)
            else:
                batch_size = 64 * max(num_gpus, 1)
            ctx = [mx.gpu(i)
                   for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
        return batch_size, ctx

    batch_size, ctx = _init_hparams(args)
    vars(args).update({'task_id': task_id})
    logger.info('Task ID : {0}, args : {1}, resources:{2}, pid:{3}'.format(task_id, args, resources, os.getpid()))

    ps_p = psutil.Process(os.getpid())
    ps_p.cpu_affinity(resources.cpu_ids)

    if 'bert' in args.model:
        net, vocab = _get_bert_pre_trained_model(args, ctx)
    elif 'lstm_lm' in args.model:  # Get LM specific model attributes
        net, vocab = _get_lm_pre_trained_model(args, ctx)

    else:
        raise ValueError('Unsupported pre-trained model type. {}  will be supported in the future.'.format(args.model))

    net.classifier = nn.Sequential()
    with net.classifier.name_scope():
        net.classifier.add(nn.Dropout(dropout=args.dropout))
        net.classifier.add(nn.Dense(args.num_classes))

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
            return args.data.transform(dataset, transform_fn)

        train_dataset = _init_dataset(args.data.train_dataset,
                                      args.data.get_transform_train_fn(args.model, vocab, args.max_sequence_length))
        val_dataset = _init_dataset(args.data.val_dataset,
                                    args.data.get_transform_val_fn(args.model, vocab, args.max_sequence_length))

        train_data = gluon.data.DataLoader(dataset=train_dataset, num_workers=args.data.num_workers,
                                           batch_sampler=args.data.get_batch_sampler(args.model, train_dataset),
                                           batchify_fn=args.data.get_batchify_fn(args.model))

        val_data = gluon.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
                                         batchify_fn=args.data.get_batchify_fn(args.model),
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
                           train_length=len(args.data.train_dataset))

    event_handlers = [early_stopping_handler, lr_handler, TextDataLoaderHandler(args.model)]

    estimator.fit(train_data=train_data, val_data=val_data, epochs=args.epochs,
                  event_handlers=event_handlers)
