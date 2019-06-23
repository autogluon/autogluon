import gluonnlp as nlp
import mxnet as mx

from autogluon.dataset.transforms import TextDataTransform
from autogluon.estimator import *
from autogluon.estimator.event_handler import DataLoaderHandler
from autogluon.scheduler.reporter import StatusReporter
from .dataset import Dataset
from .model_zoo import get_model_instances
from ...basic import autogluon_method

__all__ = ['train_text_classification']
logger = logging.getLogger(__name__)


class MeanPoolingLayer(gluon.Block):
    """
    A block for mean pooling of encoder features
    """

    def __init__(self, prefix=None, params=None):
        super(MeanPoolingLayer, self).__init__(prefix=prefix, params=params)

    def forward(self, data, valid_length):  # pylint: disable=arguments-differ
        masked_encoded = mx.ndarray.SequenceMask(data, sequence_length=valid_length, use_sequence_length=True)

        agg_state = mx.ndarray.broadcast_div(mx.ndarray.sum(masked_encoded, axis=0),
                                             mx.ndarray.expand_dims(valid_length, axis=1))

        return agg_state


class TextClassificationNet(gluon.Block):
    """
    Network for Text Classification
    """

    def __init__(self, prefix=None, params=None, num_classes=2, num_classification_layers=1, dropout=0.4):
        super(TextClassificationNet, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.embedding = None
            self.encoder = None
            self.agg_layer = MeanPoolingLayer()
            self.output = gluon.nn.Sequential()
            with self.output.name_scope():
                hidden_units = 40  # TODO Make this also a Hyperparam.
                for i in range(num_classification_layers):
                    self.output.add(gluon.nn.Dropout(rate=dropout))
                    self.output.add(gluon.nn.Dense(int(hidden_units)))
                    hidden_units = hidden_units / 2

                self.output.add(gluon.nn.Dropout(rate=dropout))
                self.output.add(gluon.nn.Dense(num_classes))

    def forward(self, data, valid_length):  # pylint: disable=arguments-differ
        encoded = self.encoder(self.embedding(data))
        agg_state = self.agg_layer(encoded, valid_length)
        out = self.output(agg_state)
        return out


class SentimentDataLoaderHandler(DataLoaderHandler):

    def batch_begin(self, estimator, *args, **kwargs):
        """
        :param estimator:
        :param batch: The batch of data
        :param ctx: The context in which to load the data.
        :param batch_axis: The batch axis about which to split the data onto multiple devices if context is passed as a list
        :return: A tuple of : (data, length), label and batch_size
        """
        batch = kwargs['batch']
        ctx = kwargs['ctx']
        batch_axis = kwargs['batch_axis'] or 0
        data = batch[0][0]
        batch_size = data.shape[0]
        lengths = batch[0][1]
        label = batch[1]
        data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=batch_axis, even_split=False)
        lengths = gluon.utils.split_and_load(lengths, ctx_list=ctx, batch_axis=batch_axis, even_split=False)
        label = gluon.utils.split_and_load(label, ctx_list=ctx, batch_axis=batch_axis, even_split=False)
        ret_data = []
        for d, length in zip(data, lengths):
            ret_data.append((d.T, length.astype(np.float32)))

        return ret_data, label, batch_size


class BERTClassifier(gluon.Block):

    def __init__(self, prefix=None, params=None, num_classes=2, num_classification_layers=1, dropout=0.4):
        super(BERTClassifier, self).__init__(prefix=prefix, params=params)
        self.pre_trained_network = None
        self.output = gluon.nn.HybridSequential()
        with self.output.name_scope():
            hidden_units = 40  # TODO Make this also a Hyperparam.
            for i in range(num_classification_layers):
                self.output.add(gluon.nn.Dropout(rate=dropout))
                self.output.add(gluon.nn.Dense(int(hidden_units)))
                hidden_units = hidden_units / 2

            self.output.add(gluon.nn.Dropout(rate=dropout))
            self.output.add(gluon.nn.Dense(num_classes))

    def forward(self, inputs, token_types, valid_length=None):  # pylint: disable=arguments-differ
        _, pooler_out = self.bert(inputs, token_types, valid_length)
        return self.output(pooler_out)


@autogluon_method
def train_text_classification(args: dict, reporter: StatusReporter, task_id: int) -> None:
    # TODO Add Estimator here.
    def _init_env():
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

    batch_size, ctx = _init_env()

    logger.info('Task ID : {0}, args : {1}'.format(task_id, args))

    mxboard_handler = args.viz if 'viz' in args else None
    if mxboard_handler is not None:
        mxboard_handler.task_id = task_id

    # Define the network and get an instance from model zoo.
    kwargs = None
    if 'bert' in args.model:
        kwargs = {'use_pooler': False,
                  'use_decoder': False,
                  'use_classifier': False,
                  'dropout_prob': 0.1}
    pre_trained_network, vocab = get_model_instances(name=args.model, pretrained=args.pretrained, ctx=ctx, **kwargs)
    # pre_trained_network is a misnomer here. This can be untrained network too.

    # fine_tune_lm(pre_trained_network) # TODO

    ## Initialize the dataset here.
    dataset_transform = TextDataTransform(vocab, transforms=[nlp.data.ClipSequence(length=500)])

    dataset = Dataset(name=args.data_name, train_path=args.train_path, val_path=args.val_path, lazy=False,
                      transform=dataset_transform, batch_size=batch_size)

    if 'bert' in args.model:
        net = BERTClassifier(num_classes=dataset.num_classes, num_classification_layers=args.dense_layers,
                             dropout=args.dropout)
        net.pre_trained_network = pre_trained_network

    else:
        net = TextClassificationNet(num_classes=dataset.num_classes, num_classification_layers=args.dense_layers,
                                    dropout=args.dropout)
        net.embedding = pre_trained_network.embedding
        net.encoder = pre_trained_network.encoder

    net.hybridize()

    logger.info('Task ID : {0}, network : {1}'.format(task_id, net))

    # define the initializer :
    # TODO : This should come from the config
    initializer = mx.init.Xavier(magnitude=2.24)
    if not args.pretrained:
        net.collect_params().initialize(init=initializer, ctx=ctx)

    else:
        net.output.initialize(init=initializer, ctx=ctx)
        net.collect_params().reset_ctx(ctx=ctx)

    # TODO : Update with search space
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    trainer = gluon.Trainer(net.collect_params(), args.optimizer, {'learning_rate': args.lr})
    estimator = Estimator(net=net, loss=loss, metrics=[mx.metric.Accuracy()], trainer=trainer, context=ctx)

    event_handlers = [SentimentDataLoaderHandler(), reporter]

    if mxboard_handler is not None:
        event_handlers.append(mxboard_handler)

    estimator.fit(train_data=dataset.train_data_loader, val_data=dataset.val_data_loader, epochs=args.epochs,
                  event_handlers=event_handlers)
