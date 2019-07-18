import gluonnlp as nlp
import mxnet as mx

from autogluon.dataset.transforms import TextDataTransform, BERTDataTransform
from autogluon.estimator import *
from autogluon.estimator import Estimator
from autogluon.scheduler.reporter import StatusReporter
from .dataset import BERTDataset, Dataset
from .event_handlers import BertDataLoaderHandler, LSTMDataLoaderHandler
from .model_zoo import get_model_instances, LMClassificationNet, BERTClassificationNet
from ...basic import autogluon_method

__all__ = ['train_text_classification']
logger = logging.getLogger(__name__)


def _init_env(args: dict):
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


def get_bert_model_attributes(args: dict, batch_size: int, ctx, num_workers):
    """
    Utility method which defines a BertModel for classification and also initializes
    dataset object compatible with BertModel.
    :param args:
    :param batch_size:
    :param ctx:
    :return: net, dataset, model_handlers
    """
    kwargs = {'use_pooler': True,
              'use_decoder': False,
              'use_classifier': False}

    pre_trained_network, vocab = get_model_instances(name=args.model, pretrained=args.pretrained, ctx=ctx, **kwargs)
    dataset_transform = BERTDataTransform(tokenizer=nlp.data.BERTTokenizer(vocab=vocab), max_seq_length=200)
    dataset = BERTDataset(name=args.data_name, train_path=args.train_path, val_path=args.val_path, lazy=False,
                          transform=dataset_transform, batch_size=batch_size, data_format=args.data_format,
                          field_indices=args.dataset.field_indices, num_workers=num_workers)

    net = BERTClassificationNet(num_classes=dataset.num_classes, num_classification_layers=args.dense_layers,
                                dropout=args.dropout)
    net.pre_trained_network = pre_trained_network

    net.hybridize(static_alloc=True)

    model_handlers = [BertDataLoaderHandler()]

    return net, dataset, model_handlers


def get_lm_model_attributes(args: dict, batch_size: int, ctx, num_workers):
    """
    Utility method which defines a Language Model for classification and also initializes
    dataset object compatible with Language Model.

    :param args:
    :param batch_size:
    :param ctx:
    :return: net, dataset, model_handlers
    """
    pre_trained_network, vocab = get_model_instances(name=args.model, pretrained=args.pretrained, ctx=ctx)
    dataset_transform = TextDataTransform(vocab, transforms=[nlp.data.ClipSequence(length=500)])

    dataset = Dataset(name=args.data_name, train_path=args.train_path, val_path=args.val_path, lazy=False,
                      transform=dataset_transform, batch_size=batch_size, data_format=args.data_format,
                      field_indices=args.dataset.field_indices, num_workers=num_workers)

    net = LMClassificationNet(num_classes=dataset.num_classes, num_classification_layers=args.dense_layers,
                              dropout=args.dropout)
    net.embedding = pre_trained_network.embedding
    net.encoder = pre_trained_network.encoder

    net.hybridize(static_alloc=True)

    model_handlers = [LSTMDataLoaderHandler()]
    return net, dataset, model_handlers


@autogluon_method
def train_text_classification(args: dict, reporter: StatusReporter, task_id: int, resources = None) -> None:
    import psutil, os
    batch_size, ctx = _init_env(args)
    args.pretrained = True
    args.task_id = task_id
    logger.info('Task ID : {0}, args : {1}, resources:{2}, pid:{3}'.format(task_id, args, resources, os.getpid()))
    ps_p = psutil.Process(os.getpid())
    ps_p.cpu_affinity(resources.cpu_ids)

    mxboard_handler = args.viz if 'viz' in args else None
    if mxboard_handler is not None:
        mxboard_handler.task_id = task_id
        mxboard_handler.trial_args = args.__dict__

    if 'bert' in args.model:  # Get bert specific model attributes
        net, dataset, model_handlers = get_bert_model_attributes(args, batch_size, ctx, resources.num_cpus)
    elif 'lstm_lm' in args.model:  # Get LM specific model attributes
        net, dataset, model_handlers = get_lm_model_attributes(args, batch_size, ctx, resources.num_cpus)

    else:
        raise ValueError('Unsupported pre-trained model type. {}  will be supported in the future.'.format(args.model))

    # pre_trained_network is a misnomer here. This can be untrained network too.

    # fine_tune_lm(pre_trained_network) # TODO

    # do not apply weight decay on LayerNorm and bias terms
    for _, v in net.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0

    logger.info('Task ID : {0}, network : {1}'.format(task_id, net))

    # define the initializer :
    # TODO : This should come from the config
    initializer = mx.init.Xavier(magnitude=2.24)
    if args.pretrained is False:
        net.collect_params().initialize(init=initializer, ctx=ctx)

    else:
        net.output.initialize(init=initializer, ctx=ctx)
        net.collect_params().reset_ctx(ctx=ctx)

    # TODO : Update with search space
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    trainer = gluon.Trainer(net.collect_params(), args.optimizer, {'learning_rate': args.lr})
    estimator: Estimator = Estimator(net=net, loss=loss, metrics=[mx.metric.Accuracy()], trainer=trainer, context=ctx)

    early_stopping_handler = EarlyStoppingHandler(monitor=estimator.train_metrics[0], mode='max')

    event_handlers = [reporter, early_stopping_handler] + model_handlers

    if mxboard_handler is not None:
        event_handlers.append(mxboard_handler)

    estimator.fit(train_data=dataset.train_data_loader, val_data=dataset.val_data_loader, epochs=args.epochs,
                  event_handlers=event_handlers)
