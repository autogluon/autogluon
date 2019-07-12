import argparse
import re
import os
import sys
import ConfigSpace as CS
import numpy as np
import autogluon as ag
from autogluon.space import Log, List, Linear, Exponential
from autogluon import named_entity_recognition as task

import logging
logging.basicConfig(level=logging.INFO)

def prepare_nets(network_list, dropout_list=None):
    if not dropout_list:
        dropout_list = [(0.0, 0.5)] * len(network_list)
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

def generate_mini_dataset(source_file_path,
                          dest_file_name,
                          column_format,
                          sentences_limit=16
                          ):
    # get the text and ner column
    text_column = sys.maxsize
    ner_column = sys.maxsize
    for column in column_format:
        if column_format[column].lower() == "text":
            text_column = column
        elif column_format[column].lower() == 'ner':
            ner_column = column
        else:
            raise ValueError("Invalid column type")

    with open(source_file_path, 'r') as ifp:
        sentence_list = []
        current_sentence = []

        for line in ifp:
            if '-DOCSTART-' in line:
                continue
            if line.startswith("#"):
                continue
            if len(line.strip()) > 0:
                fields = re.split(r'\s+', line.rstrip())
                if len(fields) > ner_column:
                    current_sentence.append([fields[text_column], fields[ner_column]])
            else:
                # the sentence was completed if an empty line occurred; flush the current sentence.
                if len(current_sentence) > 0:
                    sentence_list.append(current_sentence)
                    current_sentence = []
                    sentences_limit -= 1
                    if sentences_limit == 0:
                        break
    data_dir = os.path.join(os.path.abspath(os.sep), 'tmp', 'random_ner_data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    file_path = os.path.join(data_dir, dest_file_name)

    with open(file_path, 'w+') as fp:
        for sent in sentence_list:
            for line in sent:
                fp.write(line[0] + ' ' + line[1] + '\n')
            fp.write('\n')
    return file_path

def get_mini_dataset():
    train_path = generate_mini_dataset(source_file_path='/home/ubuntu/conll2003/train.txt',
                                       dest_file_name='train.txt',
                                       column_format={0: 'text', 3: 'ner'})
    val_path = generate_mini_dataset(source_file_path='/home/ubuntu/conll2003/valid.txt',
                                     dest_file_name='valid.txt',
                                     column_format={0: 'text', 3: 'ner'})

    dataset = task.Dataset(name='CoNLL2003',
                           train_path=train_path,
                           val_path=val_path,
                           indexes_format={0: 'text', 1: 'ner'})

    set_batch_size(dataset, 8)
    return dataset

def test_with_auto_hyp_cpu():
    """
    Test NER task on cpu.
    Run 10 trials on random samples for 1 epoch each.
    """

    dataset = get_mini_dataset()

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

    assert results.metric is not None

def test_with_fixed_hyp_cpu():
    """
    Test NER task on CPU with fixed set of hyperparameters
    """
    dataset = get_mini_dataset()

    nets = prepare_nets(['bert_24_1024_16'])

    optims = prepare_optims(['bertadam'], lr_list=[[0.00004, 0.00005]])

    stop_criterion = {
        'time_limits': 15000,
        'max_metric': 0.91,
        'max_trial_count': 1
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

    assert results.metric is not None

def test_with_auto_hyp_gpu():
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
        'max_num_gpus': 1,
        'max_num_cpus': 4,
        'max_training_epochs': 5
    }

    results = task.fit(dataset,
                       nets=nets,
                       optimizers=optims,
                       stop_criterion=stop_criterion,
                       resources_per_trial=resources_per_trial)

    assert results.metric > 0.85

def test_with_fixed_hyp_gpu():
    """
    Test NER task on GPU with fixed set of hyperparameters to assert the
    validation F1 score.
    """
    dataset = task.Dataset(name='CoNLL2003',
                           train_path='/home/ubuntu/conll2003/train.txt',
                           val_path='/home/ubuntu/conll2003/test.txt')
    dataset = set_batch_size(dataset, 8)

    nets = prepare_nets(['bert_24_1024_16'])

    optims = prepare_optims(['bertadam'], lr_list=[[0.00004, 0.00005]])

    stop_criterion = {
        'time_limits': 15000,
        'max_metric': 0.91,
        'max_trial_count': 1
    }

    resources_per_trial = {
        'max_num_gpus': 1,
        'max_num_cpus': 4,
        'max_training_epochs': 5
    }



    results = task.fit(dataset,
                       nets=nets,
                       optimizers=optims,
                       stop_criterion=stop_criterion,
                       resources_per_trial=resources_per_trial)

    assert results.metric > 0.91


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Named Entity Recognition')
    parser.add_argument('--type', type=str, default='fixed',
                        help='Options are `fixed` or `auto`')
    parser.add_argument('--machine', type=str, default='cpu',
                        help='Options are `cpu` or `gpu`')
    opt = parser.parse_args()

    if opt.machine == 'cpu' and opt.type == 'fixed':
        test_with_fixed_hyp_cpu()
    elif opt.machine == 'cpu' and opt.type == 'auto':
        test_with_auto_hyp_cpu()
    elif opt.machine == 'gpu' and opt.type == 'fixed':
        test_with_fixed_hyp_gpu()
    elif opt.machine == 'gpu' and opt.type == 'auto':
        test_with_auto_hyp_gpu()
    else:
        raise ValueError('Please provide the proper argument values. '
                         'Run with `--help` keyword argument to get the argument information')
