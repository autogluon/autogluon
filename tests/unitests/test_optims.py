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
    logger.debug('optims:')
    logger.debug(optims)
    logger.debug('search space:')
    logger.debug(optims.search_space)
    for hparam in optims.search_space.get_hyperparameters():
        logger.debug(hparam.name)
        logger.debug(type(hparam.name))
    logger.debug('Finished.')


def test_optim_strs():
    logger.debug('Start testing optimizers')
    optims = ag.Optimizers(['sgd', 'adam'])
    logger.debug('optims:')
    logger.debug(optims)
    logger.debug('search space:')
    logger.debug(optims.search_space)
    for hparam in optims.search_space.get_hyperparameters():
        logger.debug(hparam.name)
        logger.debug(type(hparam.name))
    logger.debug('Finished.')