import argparse
import ConfigSpace as CS
import numpy as np
import autogluon as ag
from autogluon.space import Log, List, Linear, Exponential
from autogluon import named_entity_recognition as task


def prepare_nets(network_list, dropout_list=None):
    if not dropout_list:
        dropout_list = [(0.0, 0.5)] * len(optim_list)
    else:
        assert len(network_list) == len(dropout_list)
    net_list = []
    for net, dropout in zip(network_list, dropout_list):
        network = task.model_zoo.get_model(net)
        setattr(network, 'hyper_params', [List('pretrained', [True]).get_hyper_param(),
                                         Linear('dropout', lower=dropout[0], upper=dropout[1]).get_hyper_param()])
        net_list.append(network)
    nets = ag.Nets(net_list)
    return nets


def prepare_optims(optim_list, lr_list=None):
    if not lr_list:
        lr_list = [(10 ** -5, 10 ** -4)] * len(optim_list)
    else:
        assert len(optim_list) == len(lr_list)
    opt_list = []
    for opt, lr in zip(optim_list, lr_list):
        optim = ag.optim.get_optim(opt)
        setattr(optim, 'hyper_params', [Log('lr', lr[0], lr[1]).get_hyper_param()])
        opt_list.append(optim)
    optims = ag.Optimizers(opt_list)
    return optims


def set_batch_size(dataset, batch_size=8):
    exp = int(np.log2(batch_size))
    cs = CS.ConfigurationSpace()
    data_hyperparams = Exponential(name='batch_size', base=2, lower_exponent=exp,
                                   upper_exponent=exp).get_hyper_param()
    cs.add_hyperparameter(data_hyperparams)
    dataset._set_search_space(cs)
    return dataset


def test_auto_cpu():
    """
    Test NER task on cpu.
    Run 10 trials on random samples for 1 epoch each.
    """
    dataset = task.Dataset(name='CoNLL2003',
                           train_path='/Users/shaabhn/Downloads/random_train.txt',
                           val_path='/Users/shaabhn/Downloads/random_test.txt',
                           indexes_format={0:'text', 1:'ner'})
    dataset = set_batch_size(dataset, 8)

    nets = prepare_nets(['bert_12_768_12', 'bert_24_1024_16'])

    optims = prepare_optims(['adam', 'bertadam'])

    stop_criterion = {
        'time_limits': 15000,
        'max_metric': 0.95,
        'max_trial_count': 10
    }
    resources_per_trial = {
        'max_num_gpus': 0,
        'max_num_cpus': 4,
        'max_training_epochs': 1
    }

    results = task.fit(dataset,
                       nets=nets,
                       optimizers=optims,
                       stop_criterion=stop_criterion,
                       resources_per_trial=resources_per_trial)


def test_auto_gpu():
    """
    Test NER task on GPU.
    Run 10 trails on CoNLL2003 for 5 epochs each and assert score.
    """

    dataset = task.Dataset(name='CoNLL2003',
                           train_path='/home/ubuntu/conll2003/train.txt',
                           val_path='/home/ubuntu/conll2003/test.txt')
    dataset = set_batch_size(dataset, 8)

    nets = prepare_nets(['bert_12_768_12', 'bert_24_1024_16'])

    optims = prepare_optims(['adam', 'bertadam'])

    stop_criterion = {
        'time_limits': 15000,
        'max_metric': 0.95,
        'max_trial_count': 10
    }
    resources_per_trial = {
        'max_num_gpus': 4,
        'max_num_cpus': 4,
        'max_training_epochs': 5
    }

    results = task.fit(dataset,
                       nets=nets,
                       optimizers=optims,
                       stop_criterion=stop_criterion,
                       resources_per_trial=resources_per_trial)

    assert results.metric > 0.85


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test autogluon NER task')
    parser.add_argument('--type', type=str, default='cpu')
    opt = parser.parse_args()
    if opt.type == 'cpu':
        test_auto_cpu()
    elif opt.type == 'gpu':
        test_auto_gpu()
    else:
        raise RuntimeError("Unknown test type")
