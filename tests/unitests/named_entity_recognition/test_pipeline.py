from unittest import mock, TestCase
import pytest
import mxnet as mx
from mxnet.gluon import loss
from mxnet.base import MXNetError
from autogluon.task.named_entity_recognition.pipeline import NERMetricHandler, NERNet
from autogluon.task.named_entity_recognition.utils import AccNer


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
