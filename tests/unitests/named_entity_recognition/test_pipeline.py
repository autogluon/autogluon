from unittest import mock, TestCase
import pytest
import numpy as np
import mxnet as mx
import gluonnlp as nlp
from mxnet import nd
from mxnet.gluon import loss
from mxnet.base import MXNetError
from autogluon.estimator import *
from autogluon.task.named_entity_recognition.pipeline import NERMetricHandler, NERNet, NEREstimator
from autogluon.task.named_entity_recognition.utils import AccNer


def _get_ner_data():
    data = []
    sample_length = 24
    for _ in range(sample_length):
        padded_text_ids = nd.ones((10,), dtype='int32')
        token_types = nd.ones((10,), dtype='int32')
        valid_length = np.array(5, dtype='int32')
        np_tag_ids = nd.ones((10,), dtype='int32')
        flag_nonnull_tag = nd.ones((10,), dtype='int32')

        data.append((padded_text_ids, token_types, valid_length,
                     np_tag_ids, flag_nonnull_tag))

    return mx.gluon.data.DataLoader(data, batch_size=8)


def _get_ner_network(ctx):
    model_kwargs = {'pretrained': True,
                    'ctx': ctx,
                    'use_pooler': False,
                    'use_decoder': False,
                    'use_classifier': False,
                    'dropout_prob': 0.1}

    pre_trained_network, _ = nlp.model.get_model(name='bert_12_768_12',
                                                 dataset_name='book_corpus_wiki_en_uncased',
                                                 **model_kwargs)
    net = pre_trained_network
    net.hybridize(static_alloc=True)

    return net


class CustomBatchHandler(BatchBegin, BatchEnd):

    def batch_begin(self, estimator, *args, **kwargs):
        print("custom batch begin")

    def batch_end(self, estimator, *args, **kwargs):
        print("custom batch end")


class CustomBatchExceptionHandler(BatchBegin, BatchEnd):

    def batch_begin(self, estimator, *args, **kwargs):
        raise RuntimeError("Testing")

    def batch_end(self, estimator, *args, **kwargs):
        print("custom batch end")


class TestNERNet(TestCase):
    def test_forward(self):
        with mock.patch('autogluon.task.named_entity_recognition.pipeline.NERNet.forward',
                        autospec=True) as MockNerNetForward:
            data = mx.nd.ones((32, 100))
            batch_size = 8
            seq_length = 10
            token_ids = mx.nd.ones((batch_size, seq_length))
            token_types = mx.nd.ones((batch_size, seq_length))
            valid_length = mx.nd.ones((batch_size,))
            MockNerNetForward.return_value = data
            net = NERNet()
            assert isinstance(net.forward(token_ids, token_types, valid_length), mx.nd.NDArray)


@pytest.mark.serial
def test_ner_metric_handler():
    train_metrics = [AccNer()]
    metric_handler = NERMetricHandler(train_metrics=train_metrics)
    kwargs = {}
    kwargs['pred'] = mx.nd.ones((4, 10, 16))
    kwargs['label'] = mx.nd.ones((4, 10))
    kwargs['flag_nonnull_tag'] = mx.nd.ones((4, 10))
    kwargs['loss'] = loss.SoftmaxCrossEntropyLoss()
    metric_handler.batch_end(estimator=None, **kwargs)

    with pytest.raises(MXNetError):
        kwargs['pred'] = mx.nd.ones((4, 20, 16))
        metric_handler.batch_end(estimator=None, **kwargs)


@pytest.mark.serial
def test_train():
    ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()
    data = _get_ner_data()
    net = _get_ner_network(ctx)

    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    acc = mx.metric.Accuracy()
    ner_est = NEREstimator(net, loss)

    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.0005})
    est_ref = Estimator(net=net, loss=loss, metrics=acc,
                        trainer=trainer, context=ctx)

    # Without batch event handler
    ner_est.train(train_data=data,
                  estimator_ref=est_ref,
                  batch_begin=[],
                  batch_end=[])

    # With custom batch event handler
    cbh = CustomBatchHandler()
    ner_est.train(train_data=data,
                  estimator_ref=est_ref,
                  batch_begin=[cbh],
                  batch_end=[cbh])

    # With custom batch event handler that raise RuntimeError
    cbeh = CustomBatchExceptionHandler()
    with pytest.raises(RuntimeError):
        ner_est.train(train_data=data,
                      estimator_ref=est_ref,
                      batch_begin=[cbeh],
                      batch_end=[cbh])
