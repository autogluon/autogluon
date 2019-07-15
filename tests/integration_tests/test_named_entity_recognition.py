import re
import os
import sys
import unittest
import numpy as np
import mxnet as mx
import ConfigSpace as CS

from autogluon.space import Exponential
from autogluon import named_entity_recognition as task

import logging

logging.basicConfig(level=logging.INFO)


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
                                     column_format={0: 'text', 3: 'ner'},
                                     sentences_limit=8)

    dataset = task.Dataset(name='CoNLL2003',
                           train_path=train_path,
                           val_path=val_path,
                           indexes_format={0: 'text', 1: 'ner'})

    set_batch_size(dataset, 8)
    return dataset


def test_with_auto_hyp_cpu():
    """
    Test NER task on cpu with auto hyperparameter search.
    Run 10 trials on random samples for 1 epoch each.
    """

    dataset = get_mini_dataset()

    nets = task.utils.prepare_nets(['bert_12_768_12', 'bert_24_1024_16'])

    optims = task.utils.prepare_optims(['adam', 'bertadam'])

    stop_criterion = {
        'time_limits': 15000,
        'max_metric': 0.95,
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


def test_with_fixed_hyp_cpu():
    """
    Test NER task on CPU with fixed set of hyperparameters
    """
    dataset = get_mini_dataset()

    nets = task.utils.prepare_nets(['bert_24_1024_16'])

    optims = task.utils.prepare_optims(['bertadam'], lr_list=[[0.00004, 0.00005]])

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


@unittest.skipIf(mx.context.num_gpus() < 1, "skip if no GPU found")
def test_with_auto_hyp_gpu():
    """
    Test NER task on GPU with auto hyperparameter search.
    Run 10 trails on CoNLL2003 for 5 epochs each and assert score.
    """

    dataset = task.Dataset(name='CoNLL2003',
                           train_path='/home/ubuntu/conll2003/train.txt',
                           val_path='/home/ubuntu/conll2003/test.txt')
    dataset = set_batch_size(dataset, 8)

    nets = task.utils.prepare_nets(['bert_12_768_12', 'bert_24_1024_16'])

    optims = task.utils.prepare_optims(['adam', 'bertadam'])

    stop_criterion = {
        'time_limits': 15000,
        'max_metric': 0.95,
        'max_trial_count': 1
    }
    resources_per_trial = {
        'max_num_gpus': 1,
        'max_num_cpus': 4,
        'max_training_epochs': 1
    }

    results = task.fit(dataset,
                       nets=nets,
                       optimizers=optims,
                       stop_criterion=stop_criterion,
                       resources_per_trial=resources_per_trial)

    assert results.metric > 0.85


@unittest.skipIf(mx.context.num_gpus() < 1, "skip if no GPU found")
def test_with_fixed_hyp_gpu():
    """
    Test NER task on GPU with fixed set of hyperparameters to assert the
    validation F1 score.
    """
    dataset = task.Dataset(name='CoNLL2003',
                           train_path='/home/ubuntu/conll2003/train.txt',
                           val_path='/home/ubuntu/conll2003/test.txt')
    dataset = set_batch_size(dataset, 8)

    nets = task.utils.prepare_nets(['bert_24_1024_16'])

    optims = task.utils.prepare_optims(['bertadam'], lr_list=[[0.00004, 0.00005]])

    stop_criterion = {
        'time_limits': 15000,
        'max_metric': 0.91,
        'max_trial_count': 1
    }

    resources_per_trial = {
        'max_num_gpus': 1,
        'max_num_cpus': 4,
        'max_training_epochs': 1
    }

    results = task.fit(dataset,
                       nets=nets,
                       optimizers=optims,
                       stop_criterion=stop_criterion,
                       resources_per_trial=resources_per_trial)

    assert results.metric > 0.90


if __name__ == '__main__':
    import nose

    nose.runmodule()
