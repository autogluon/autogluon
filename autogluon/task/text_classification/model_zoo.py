from typing import AnyStr

import gluonnlp as nlp
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import Block

from autogluon.network import autogluon_nets, autogluon_net_instances, Net

__all__ = ['get_model_instances', 'get_model', 'models', 'LMClassifier', 'BERTClassifier']

models = ['standard_lstm_lm_200',
          'standard_lstm_lm_650',
          'standard_lstm_lm_1500',
          'awd_lstm_lm_1150',
          'awd_lstm_lm_600',
          'elmo_2x1024_128_2048cnn_1xhighway',
          'elmo_2x2048_256_2048cnn_1xhighway',
          'elmo_2x4096_512_2048cnn_2xhighway',
          'bert_12_768_12',
          'bert_24_1024_16']


@autogluon_net_instances
def get_model_instances(name: AnyStr,
                        dataset_name: AnyStr = 'wikitext-2', **kwargs) -> (Block, nlp.vocab):
    """
    Parameters
    ----------
    name : str
        Name of the model.
    dataset_name : str or None, default 'wikitext-2'.
        The dataset name on which the pre-trained model is trained.
        For language model, options are 'wikitext-2'.
        For ELMo, Options are 'gbw' and '5bw'.
        'gbw' represents 1 Billion Word Language Model Benchmark
        http://www.statmt.org/lm-benchmark/;
        '5bw' represents a dataset of 5.5B tokens consisting of
        Wikipedia (1.9B) and all of the monolingual news crawl data from WMT 2008-2012 (3.6B).
        If specified, then the returned vocabulary is extracted from
        the training set of the dataset.
        If None, then vocab is required, for specifying embedding weight size, and is directly
        returned.
    vocab : gluonnlp.Vocab or None, default None
        Vocabulary object to be used with the language model.
        Required when dataset_name is not specified.
        None Vocabulary object is required with the ELMo model.
    pretrained : bool, default False
        Whether to load the pre-trained weights for model.
    ctx : Context, default CPU
        The context in which to load the pre-trained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    gluon.Block, gluonnlp.Vocab, (optional) gluonnlp.Vocab
    """
    if name is None:
        raise ValueError("model name cannot be passed as none for the model")
    name = name.lower()

    if 'bert' in name:
        # Currently the dataset for BERT is book corpus wiki only on gluon model zoo
        dataset_name = 'book_corpus_wiki_en_uncased'

    if name not in models:
        err_str = '{} is not among the following model list: \n\t'.format(name)
        raise ValueError(err_str)
    return nlp.model.get_model(name=name, dataset_name=dataset_name, **kwargs)


@autogluon_nets
def get_model(name: AnyStr, **kwargs) -> Net:
    """Returns a network with search space by name

        Parameters
        ----------
        name : str
            Name of the model.
        pretrained : bool or str
            Boolean value controls whether to load the default pretrained weights for model.
            String value represents the hashtag for a certain version of pretrained weights.
        classes : int
            Number of classes for the output layer.
        ctx : Context, default CPU
            The context in which to load the pretrained weights.
        root : str, default '~/.mxnet/models'
            Location for keeping the model parameters.

        Returns
        -------
        Net
            The model with search space.
        """
    name = name.lower()
    if name not in models:
        err_str = '{} is not among the following model list:\n\t'.format(name)
        err_str += '%s' % ('\n\t'.join(sorted(models)))
        raise ValueError(err_str)
    net = Net(name)
    return net


# TODO (ghaipiyu): add more models using method

@autogluon_nets
def standard_lstm_lm_200(**kwargs):
    pass


@autogluon_nets
def standard_lstm_lm_650(**kwargs):
    pass


@autogluon_nets
def standard_lstm_lm_1500(**kwargs):
    pass


@autogluon_nets
def awd_lstm_lm_1150(**kwargs):
    pass


@autogluon_nets
def awd_lstm_lm_600(**kwargs):
    pass


@autogluon_nets
def elmo_2x1024_128_2048cnn_1xhighway(**kwargs):
    pass


@autogluon_nets
def elmo_2x2048_256_2048cnn_1xhighway(**kwargs):
    pass


@autogluon_nets
def elmo_2x4096_512_2048cnn_2xhighway(**kwargs):
    pass


@autogluon_nets
def bert_12_768_12(**kwargs):
    pass


@autogluon_nets
def bert_24_1024_16(**kwargs):
    pass


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

    def forward(self, data, valid_length):  # pylint: disable=arguments-differ
        encoded = self.encoder(self.embedding(data))
        # Add mean pooling to the output of the LSTM Layers
        masked_encoded = mx.ndarray.SequenceMask(encoded, sequence_length=valid_length, use_sequence_length=True)
        agg_state = mx.ndarray.broadcast_div(mx.ndarray.sum(masked_encoded, axis=0),
                                             mx.ndarray.expand_dims(valid_length, axis=1))
        out = self.classifier(agg_state)
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
        with self.name_scope():
            self.classifier = gluon.nn.HybridSequential(prefix=prefix)
            if dropout:
                self.classifier.add(gluon.nn.Dropout(rate=dropout))
            self.classifier.add(gluon.nn.Dense(units=num_classes))

    def forward(self, inputs, token_types, valid_length=None):  # pylint: disable=arguments-differ
        _, pooler_out = self.bert(inputs, token_types, valid_length)
        return self.classifier(pooler_out)
