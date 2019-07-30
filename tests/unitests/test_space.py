from __future__ import print_function

import pytest
import logging

import autogluon as ag

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


@pytest.mark.serial
def test_list_space():
    logger.debug('Start testing list space')
    list_space = ag.space.List('listspace', ['0',
                                             '1',
                                             '2'])
    print(list_space)
    logger.debug('Finished.')


@pytest.mark.serial
def test_linear_space():
    logger.debug('Start testing linear space')
    linear_space = ag.space.Linear('linspace', 0, 10)
    print(linear_space)
    logger.debug('Finished.')


@pytest.mark.serial
def test_log_space():
    logger.debug('Start testing log space')
    log_space = ag.space.Log('logspace', 10**-10, 10**-1)
    print(log_space)
    logger.debug('Finished.')
