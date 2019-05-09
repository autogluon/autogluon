from __future__ import print_function

import pytest
import logging

import autogluon as ag


@pytest.mark.serial
def test_dataset():
    train_dataset, valid_dataset = ag.Dataset('./CIFAR10/train', './CIFAR10/valid')
    logging.info(train_dataset)
    logging.info(valid_dataset)
    pass