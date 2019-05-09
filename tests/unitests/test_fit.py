from __future__ import print_function

import pytest
import logging

import autogluon as ag


@pytest.mark.serial
def test_fit():
    train_dataset, valid_dataset = ag.Dataset('./CIFAR10/train', './CIFAR10/valid')
    models = ag.fit(train_dataset)
    logging.info('trials results:')
    logging.info(models[0])