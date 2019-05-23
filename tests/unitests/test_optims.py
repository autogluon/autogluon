from __future__ import print_function

import pytest
import logging

import autogluon as ag

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


@pytest.mark.serial
def test_optims():
    logger.debug('Start testing optimizers')
    optims = ag.Optimizers([ag.optim.optims.get_optim('sgd'),
                            ag.optim.optims.get_optim('adam')])
    print('optims:')
    print(optims)
    print('search space:')
    print(optims.search_space)
    for hparam in optims.search_space.get_hyperparameters():
        print(hparam.name)
        print(type(hparam.name))
    logger.debug('Finished.')


def test_optim_strs():
    logger.debug('Start testing optimizers')
    optims = ag.Optimizers(['sgd', 'adam'])
    print('optims:')
    print(optims)
    print('search space:')
    print(optims.search_space)
    for hparam in optims.search_space.get_hyperparameters():
        print(hparam.name)
        print(type(hparam.name))
    logger.debug('Finished.')