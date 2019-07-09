from __future__ import print_function

import pytest
import gluonnlp as nlp
from autogluon.task.named_entity_recognition.model_zoo import get_model, get_model_instances


@pytest.mark.serial
def test_get_model():
    with pytest.raises(ValueError):
        _ = get_model(name='Incorrect_name')

    net = get_model(name='bert_12_768_12')
    assert net.name == 'bert_12_768_12'
    assert net.hyper_params is not None


@pytest.mark.serial
def test_get_model_instances():
    with pytest.raises(ValueError):
        _, _ = get_model_instances(name=None)

    with pytest.raises(ValueError):
        _, _ = get_model_instances(name='incorrect_model_name')

    _, vocab = get_model_instances(name='standard_lstm_lm_200')
    assert isinstance(vocab, nlp.Vocab)
