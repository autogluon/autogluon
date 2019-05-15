from __future__ import print_function

import pytest
import logging

import autogluon as ag

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


@pytest.mark.serial
def test_dataset():
    logger.debug('Start testing dataset.')
    train_dataset, valid_dataset = ag.Dataset('./CIFAR10/train', './CIFAR10/valid')
    print(train_dataset)
    print(valid_dataset)
    logger.debug('Finished.')
