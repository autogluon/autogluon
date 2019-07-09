import logging
import numpy as np
from mxnet import autograd, gluon, cpu, gpu, init
import gluonnlp as nlp
from autogluon.estimator.event_handler import LoggingHandler, LRHandler
from autogluon.estimator.event_handler import ValidationHandler, MetricHandler
from autogluon.estimator.event_handler import EarlyStoppingHandler
from autogluon.estimator.estimator import Estimator
from autogluon.scheduler.reporter import StatusReporter

from .utils import AccNer, F1Ner, convert_arrays_to_text
from .dataset import Dataset
from .model_zoo import get_model_instances
from ...basic import autogluon_method

__all__ = ['train_named_entity_recognizer']
logger = logging.getLogger(__name__)


class NERNet(gluon.Block):
    """
    Network for Named Entity Recognition
    """

    def __init__(self, prefix=None, params=None, num_classes=2, dropout=0.1):
        super(NERNet, self).__init__(prefix=prefix, params=params)
        self.backbone = None
        self.output = gluon.nn.HybridSequential()
        with self.output.name_scope():
            self.output.add(gluon.nn.Dropout(rate=dropout))
            self.output.add(gluon.nn.Dense(num_classes, flatten=False))

    def forward(self, token_ids, token_types, valid_length):
        """Generate an unnormalized score for the tag of each token
        Parameters
        ----------
        token_ids: NDArray, shape (batch_size, seq_length)
            ID of tokens in sentences
            See `input` of `glounnlp.model.BERTModel`
        token_types: NDArray, shape (batch_size, seq_length)
            See `gluonnlp.model.BERTModel`
        valid_length: NDArray, shape (batch_size,)
            See `gluonnlp.model.BERTModel`
        Returns
        -------
        NDArray, shape (batch_size, seq_length, num_tag_types):
            Unnormalized prediction scores for each tag on each position.
        """
        output = self.output(self.backbone(token_ids, token_types, valid_length))
        return output


class NEREstimator(Estimator):
    """NER Estimator class that overrides the train method"""
    def __init__(self, net,
                 loss,
                 metrics=None,
                 initializer=None,
                 trainer=None,
                 context=None,
                 grad_clip=True):

        super(NEREstimator, self).__init__(net,
                                           loss,
                                           metrics,
                                           initializer,
                                           trainer,
                                           context)

        self.params = [p for p in self.net.collect_params().values() if p.grad_req != 'null']
        self.grad_clip = grad_clip

    def train(self,
              train_data,
              estimator_ref,
              batch_begin,
              batch_end,
              batch_axis=0):
        for batch in train_data:
            text_ids, token_types, valid_length, tag_ids, flag_nonnull_tag = \
                [gluon.utils.split_and_load(x.astype(np.float32),
                                            ctx_list=self.context,
                                            even_split=False)
                 for x in batch]
            batch_size = batch[0].shape[0]

            # batch begin
            for handler in batch_begin:
                handler.batch_begin(estimator_ref, batch=batch)

            with autograd.record():
                out = [self.net(ti, tt, vl) for ti, tt, vl
                       in zip(text_ids, token_types, valid_length)]
                loss_value = [self.loss[0](o, ti, fl.expand_dims(axis=2))
                              for o, ti, fl in zip(out, tag_ids, flag_nonnull_tag)]

            for l in loss_value:
                l.backward()

            if self.grad_clip:
                nlp.utils.clip_grad_global_norm(self.params, 1)

            self.trainer.step(batch_size)

            # batch end
            batch_end_result = []
            for handler in batch_end:
                batch_end_result.append(handler.batch_end(estimator_ref, batch=batch,
                                                          pred=out, label=tag_ids,
                                                          flag_nonnull_tag=flag_nonnull_tag,
                                                          loss=loss_value))
            # if any handler signaled to stop
            if any(batch_end_result):
                break


class NERMetricHandler(MetricHandler):
    """Metric update Handler for NER"""
    def __init__(self, train_metrics):
        super(NERMetricHandler, self).__init__(train_metrics)

    def batch_end(self, estimator, *args, **kwargs):
        pred = kwargs['pred']
        label = kwargs['label']
        flag_nonnull_tag = kwargs['flag_nonnull_tag']
        loss = kwargs['loss']
        for metric in self.train_metrics:
            if isinstance(metric, gluon.loss.Loss):
                # metric wrapper for loss values
                metric.update(0, loss)
            elif isinstance(metric, AccNer):
                metric.update(label, pred, flag_nonnull_tag)


@autogluon_method
def train_named_entity_recognizer(args: dict, reporter: StatusReporter, task_id: int) -> None:
    """Train method for NER"""
    def _init_env():
        if hasattr(args, 'batch_size') and hasattr(args, 'num_gpus'):
            batch_size = args.batch_size * max(args.num_gpus, 1)
            ctx = [gpu(i)
                   for i in range(args.num_gpus)] if args.num_gpus > 0 else [cpu()]
        else:
            if hasattr(args, 'num_gpus'):
                num_gpus = args.num_gpus
            else:
                num_gpus = 0
            if hasattr(args, 'batch_size'):
                batch_size = args.batch_size * max(num_gpus, 1)
            else:
                batch_size = 8 * max(num_gpus, 1)
            ctx = [gpu(i)
                   for i in range(num_gpus)] if num_gpus > 0 else [cpu()]
        return batch_size, ctx

    batch_size, ctx = _init_env()

    logger.info('Task ID : %d, args : %s', task_id, args)

    # Define the network and get an instance from model zoo.
    model_kwargs = {'pretrained': args.pretrained,
                    'ctx': ctx,
                    'use_pooler': False,
                    'use_decoder': False,
                    'use_classifier': False,
                    'dropout_prob': 0.1}
    pre_trained_network, vocab = get_model_instances(name=args.model,
                                                     dataset_name='book_corpus_wiki_en_cased',
                                                     **model_kwargs)

    ## Initialize the dataset here.
    dataset = Dataset(name=args.data_name, train_path=args.train_path, val_path=args.val_path,
                      lazy=False, vocab=vocab, indexes_format=args.indexes_format,
                      batch_size=batch_size, max_sequence_length=args.max_sequence_length)

    net = NERNet(num_classes=dataset.num_classes, dropout=args.dropout)
    net.backbone = pre_trained_network

    logger.info('Task ID : %d, network : %s', task_id, net)

    # define the initializer :
    # TODO : This should come from the config
    initializer = init.Normal(sigma=0.02)
    if not args.pretrained:
        net.collect_params().initialize(init=initializer, ctx=ctx)

    else:
        net.output.initialize(init=initializer, ctx=ctx)

    net.hybridize(static_alloc=True)

    # do not apply weight decay on LayerNorm and bias terms
    for _, v in net.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0

    # TODO : Update with search space
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    trainer = gluon.Trainer(net.collect_params(), args.optimizer, {'learning_rate': args.lr})

    lr_handler = LRHandler(warmup_ratio=0.1,
                           batch_size=batch_size,
                           num_epochs=args.epochs,
                           train_length=len(dataset.train_dataset))

    train_metrics = [AccNer()]
    val_metrics = [F1Ner()]
    for metric in val_metrics:
        metric.name = "validation " + metric.name

    metric_handler = NERMetricHandler(train_metrics)

    def eval_ner(estimator, val_data, val_metrics):
        if not isinstance(val_data, gluon.data.DataLoader):
            raise ValueError("Estimator only support input as Gluon DataLoader. Alternatively, you "
                             "can transform your DataIter or any NDArray into Gluon DataLoader. "
                             "Refer to gluon.data.dataloader")

        for metric in val_metrics:
            metric.reset()

        predictions = []
        for _, batch in enumerate(val_data):

            text_ids, token_types, valid_length, tag_ids, _ = [
                x.astype(np.float32).as_in_context(ctx[0]) for x in batch]
            out = net(text_ids, token_types, valid_length)

            # convert results to numpy arrays for easier access
            np_text_ids = text_ids.astype('int32').asnumpy()
            np_pred_tags = out.argmax(axis=-1).asnumpy()
            np_valid_length = valid_length.astype('int32').asnumpy()
            np_true_tags = tag_ids.asnumpy()

            predictions += convert_arrays_to_text(vocab, dataset.tag_vocab, np_text_ids,
                                                  np_true_tags, np_pred_tags, np_valid_length)

        all_true_tags = [[entry.true_tag for entry in entries] for entries in predictions]
        all_pred_tags = [[entry.pred_tag for entry in entries] for entries in predictions]

        # update metrics
        for metric in val_metrics:
            if isinstance(metric, F1Ner):
                metric.update(all_true_tags, all_pred_tags)

    val_handler = ValidationHandler(val_data=dataset.val_dataloader,
                                    eval_fn=eval_ner,
                                    val_metrics=val_metrics)

    log_handler = LoggingHandler(train_metrics=train_metrics,
                                 val_metrics=val_metrics,
                                 verbose=LoggingHandler.LOG_PER_EPOCH)

    stop_handler = EarlyStoppingHandler(monitor=val_metrics[0],
                                        mode='max')

    estimator = NEREstimator(net=net, loss=loss, metrics=train_metrics,
                             trainer=trainer, context=ctx)
    estimator.val_metrics = val_metrics

    estimator.fit(train_data=dataset.train_dataloader,
                  val_data=dataset.val_dataloader,
                  epochs=args.epochs,
                  event_handlers=[lr_handler, metric_handler, val_handler, stop_handler, log_handler, reporter])
