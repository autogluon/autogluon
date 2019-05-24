from typing import AnyStr

import gluonnlp as nlp
from autogluon.network import autogluon_nets, autogluon_net_instances, Net
from mxnet.gluon import Block

__all__ = ['get_model_instances', 'get_model', 'models']

models = ['standard_lstm_lm_200',
          'standard_lstm_lm_650',
          'standard_lstm_lm_1500',
          'awd_lstm_lm_1150',
          'awd_lstm_lm_600',
          'big_rnn_lm_2048_512',
          'elmo_2x1024_128_2048cnn_1xhighway',
          'elmo_2x2048_256_2048cnn_1xhighway',
          'elmo_2x4096_512_2048cnn_2xhighway',
          'transformer_en_de_512',
          'bert_12_768_12',
          'bert_24_1024_16']


@autogluon_net_instances
def get_model_instances(name: AnyStr,
                        pretrained_dataset='wikitext-2', **kwargs) -> (Block, nlp.vocab):
    """
    Parameters
    ----------
    name : str
        Name of the model.
    pretrained_dataset : str or None, default 'wikitext-2'.
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

    if name not in models:
        err_str = '{} is not among the following model list: \n\t'.format(name)
        raise ValueError(err_str)
    return nlp.model.get_model(name=name, dataset_name=pretrained_dataset, **kwargs)


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
def big_rnn_lm_2048_512(**kwargs):
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
def transformer_en_de_512(**kwargs):
    pass


@autogluon_nets
def bert_12_768_12(**kwargs):
    pass


@autogluon_nets
def bert_24_1024_16(**kwargs):
    pass
