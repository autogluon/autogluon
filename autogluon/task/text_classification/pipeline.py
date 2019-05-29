import mxnet as mx
from autogluon.estimator import *
from autogluon.scheduler.reporter import StatusReporter
from mxnet import gluon

from .dataset import Dataset
from .model_zoo import get_model_instances
from ...basic import autogluon_method

__all__ = ['train_text_classification']


class MeanPoolingLayer(gluon.Block):
    """
    A block for mean pooling of encoder features
    """

    def __init__(self, prefix=None, params=None):
        super(MeanPoolingLayer, self).__init__(prefix=prefix, params=params)

    def forward(self, data, sequence_length):  # pylint: disable=arguments-differ
        masked_encoded = mx.ndarray.SequenceMask(data, sequence_length=sequence_length, use_sequence_length=True)

        agg_state = mx.ndarray.broadcast_div(mx.ndarray.sum(masked_encoded, axis=0),
                                             mx.ndarray.expand_dims(sequence_length, axis=1))

        return agg_state


class TextClassificationNet(gluon.Block):
    """
    Network for Text Classification
    """

    def __init__(self, prefix=None, params=None, num_classes=2):
        super(TextClassificationNet, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.embedding = None
            self.encoder = None
            self.agg_layer = MeanPoolingLayer()
            self.output = gluon.nn.Sequential()
            with self.output.name_scope():
                self.output.add(gluon.nn.Dropout(rate=0.4))
                self.output.add(gluon.nn.Dense(num_classes))

    def forward(self, data, valid_length):  # pylint: disable=arguments-differ
        encoded = self.encoder(self.embedding(data))
        agg_state = self.agg_layer(encoded, valid_length)
        out = self.output(agg_state)
        return out


@autogluon_method
def train_text_classification(args: dict, reporter: StatusReporter) -> None:
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

    # Define the network and get an instance from model zoo.
    pre_trained_network, vocab = get_model_instances(name=args.model, pretrained=args.pretrained, ctx=ctx)
    # pre_trained_network is a misnomer here. This can be untrained network too.

    ## Initialize the dataset here.
    dataset = Dataset(name=args.data_name, train_path=args.train_path, val_path=args.val_path, lazy=False, vocab=vocab,
                      batch_size=batch_size)

    net = TextClassificationNet(num_classes=dataset.num_classes)
    net.embedding = pre_trained_network.embedding
    net.encoder = pre_trained_network.encoder

    net.hybridize()

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

    trainer = gluon.Trainer(net.collect_params(), 'ftml', {'learning_rate': args.lr})
    estimator = Estimator(net=net, loss=loss, metrics=[mx.metric.Accuracy()], trainer=trainer, context=ctx)

    estimator.fit(train_data=dataset.train_data_loader, val_data=dataset.val_data_loader, epochs=args.epochs)

    print(estimator.val_metrics)  # TODO: Add callback here

    # TODO : Add More event handlers here to plug the results
