from __future__ import print_function

import pytest
import logging

import autogluon as ag

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


@pytest.mark.serial
def test_autogluon_fit():
    logger.debug('Start testing autogluon fit')
    dataset = ag.Dataset('./CIFAR10/train', './CIFAR10/valid')
    models = ag.fit(dataset)
    logger.debug('Finished.')


@pytest.mark.serial
def test_backend_ray_fit():
    logger.debug('Start testing ray fit')
    dataset = ag.Dataset('./CIFAR10/train', './CIFAR10/valid')
    models = ag.fit(dataset, backend='ray')
    print('trials results:')
    print(models[0])
    logger.debug('Finished.')
