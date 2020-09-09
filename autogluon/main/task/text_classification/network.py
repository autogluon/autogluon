from typing import AnyStr

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import Block, HybridBlock, nn
from .dataset import *


__all__ = ['get_network', 'LMClassifier', 'BERTClassifier', 'RoBERTaClassifier']

def get_network(bert, class_labels, use_roberta=False):
    do_regression = not class_labels
    num_classes = 1 if do_regression else len(class_labels)
    # reuse the BERTClassifier class with num_classes=1 for regression
    if use_roberta:
        model = RoBERTaClassifier(bert, dropout=0.0, num_classes=num_classes)
    else:
        model = BERTClassifier(bert, dropout=0.1, num_classes=num_classes)
    return model


class LMClassifier(gluon.Block):
    """
    Network for Text Classification which uses a pre-trained language model.
    This works with  standard_lstm_lm_200, standard_lstm_lm_650, standard_lstm_lm_1500, awd_lstm_lm_1150, awd_lstm_lm_600
    """

    def __init__(self, prefix=None, params=None, embedding=None):
        super(LMClassifier, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.embedding = embedding
            self.encoder = None
            self.classifier = None
            self.pool_out = None

    def forward(self, data, valid_length):  # pylint: disable=arguments-differ
        encoded = self.encoder(self.embedding(data))
        # Add mean pooling to the output of the LSTM Layers
        masked_encoded = mx.ndarray.SequenceMask(encoded, sequence_length=valid_length, use_sequence_length=True)
        self.pool_out = mx.ndarray.broadcast_div(mx.ndarray.sum(masked_encoded, axis=0),
                                             mx.ndarray.expand_dims(valid_length, axis=1))
        out = self.classifier(self.pool_out)
        return out


class BERTClassifier(gluon.Block):
    """
    Network for Text Classification which uses a BERT pre-trained model.
    This works with  bert_12_768_12, bert_24_1024_16.
    Adapted from https://github.com/dmlc/gluon-nlp/blob/master/scripts/bert/model/classification.py#L76
    """

    def __init__(self, bert, num_classes=2, dropout=0.0, prefix=None, params=None):
        super(BERTClassifier, self).__init__(prefix=prefix, params=params)
        self.bert = bert
        self.pool_out = None
        with self.name_scope():
            self.classifier = gluon.nn.HybridSequential(prefix=prefix)
            if dropout:
                self.classifier.add(gluon.nn.Dropout(rate=dropout))
            self.classifier.add(gluon.nn.Dense(units=num_classes))

    def forward(self, inputs, token_types, valid_length=None):  # pylint: disable=arguments-differ
        _, self.pooler_out = self.bert(inputs, token_types, valid_length)
        return self.classifier(self.pooler_out)

class RoBERTaClassifier(HybridBlock):
    """Model for sentence (pair) classification task with BERT.
    The model feeds token ids and token type ids into BERT to get the
    pooled BERT sequence representation, then apply a Dense layer for
    classification.
    Parameters
    ----------
    roberta: RoBERTaModel
        The RoBERTa model.
    num_classes : int, default is 2
        The number of target classes.
    dropout : float or None, default 0.0.
        Dropout probability for the bert output.
    prefix : str or None
        See document of `mx.gluon.Block`.
    params : ParameterDict or None
        See document of `mx.gluon.Block`.
    Inputs:
        - **inputs**: input sequence tensor, shape (batch_size, seq_length)
        - **valid_length**: optional tensor of input sequence valid lengths.
            Shape (batch_size, num_classes).
    Outputs:
        - **output**: Regression output, shape (batch_size, num_classes)
    """

    def __init__(self, roberta, num_classes=2, dropout=0.0,
                 prefix=None, params=None):
        super(RoBERTaClassifier, self).__init__(prefix=prefix, params=params)
        self.roberta = roberta
        self._units = roberta._units
        with self.name_scope():
            self.classifier = nn.HybridSequential(prefix=prefix)
            if dropout:
                self.classifier.add(nn.Dropout(rate=dropout))
            self.classifier.add(nn.Dense(units=self._units, activation='tanh'))
            if dropout:
                self.classifier.add(nn.Dropout(rate=dropout))
            self.classifier.add(nn.Dense(units=num_classes))

    def __call__(self, inputs, valid_length=None):
        # pylint: disable=dangerous-default-value, arguments-differ
        """Generate the unnormalized score for the given the input sequences.
        Parameters
        ----------
        inputs : NDArray or Symbol, shape (batch_size, seq_length)
            Input words for the sequences.
        valid_length : NDArray or Symbol, or None, shape (batch_size)
            Valid length of the sequence. This is used to mask the padded tokens.
        Returns
        -------
        outputs : NDArray or Symbol
            Shape (batch_size, num_classes)
        """
        return super(RoBERTaClassifier, self).__call__(inputs, valid_length)

    def hybrid_forward(self, F, inputs, valid_length=None):
        # pylint: disable=arguments-differ
        """Generate the unnormalized score for the given the input sequences.
        Parameters
        ----------
        inputs : NDArray or Symbol, shape (batch_size, seq_length)
            Input words for the sequences.
        valid_length : NDArray or Symbol, or None, shape (batch_size)
            Valid length of the sequence. This is used to mask the padded tokens.
        Returns
        -------
        outputs : NDArray or Symbol
            Shape (batch_size, num_classes)
        """
        seq_out = self.roberta(inputs, valid_length)
        assert not isinstance(seq_out, (tuple, list)), 'Expected one output from RoBERTaModel'
        outputs = seq_out.slice(begin=(0, 0, 0), end=(None, 1, None))
        outputs = outputs.reshape(shape=(-1, self._units))
        return self.classifier(outputs)
