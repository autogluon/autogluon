import pytest
import logging

import autogluon as ag

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


@pytest.mark.serial
def test_dataset():
    logger.debug('Start testing dataset.')
    dataset = ag.task.Dataset(name='cifar10')
    logger.debug(dataset)
    logger.debug('Finished.')
