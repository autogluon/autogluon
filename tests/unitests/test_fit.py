import pytest
import logging

import autogluon as ag

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


@pytest.mark.serial
def test_autogluon_fit():
    logger.debug('Start testing autogluon fit')
    dataset = ag.task.Dataset(name='cifar10')
    results = ag.fit(dataset)
    logger.debug(results)
    logger.debug('Finished.')


@pytest.mark.serial
def test_backend_ray_fit():
    logger.debug('Start testing ray fit')
    dataset = ag.task.Dataset(name='cifar10')
    results = ag.fit(dataset, backend='ray')
    logger.debug(results)
    logger.debug('Finished.')
