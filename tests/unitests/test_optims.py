import pytest
import logging

import autogluon as ag

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


@pytest.mark.serial
def test_optims():
    logger.debug('Start testing optimizers')
    optims = ag.Optimizers([ag.optimizer.optimizers.get_optim('sgd'),
                            ag.optimizer.optimizers.get_optim('adam')])
    logger.debug('optims:')
    logger.debug(optims)
    logger.debug('search space:')
    logger.debug(optims.search_space)
    for hparam in optims.search_space.get_hyperparameters():
        logger.debug(hparam.name)
        logger.debug(type(hparam.name))
    logger.debug('Finished.')


@pytest.mark.serial
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


@pytest.mark.serial
def test_custom_optim_range():
    logger.debug('Start custom optimizer range')
    opt = ag.optimizers.Adam(lr=ag.space.Log('lr', 10 ** -2, 10 ** -1),
                             wd=ag.space.Log('wd', 10 ** -6, 10 ** -2))
    logger.debug(opt.hyper_params)
    logger.debug('Finished.')
