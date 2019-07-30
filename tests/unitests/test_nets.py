import pytest
import logging

import autogluon as ag

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


@pytest.mark.serial
def test_nets():
    logger.debug('Start testing nets')
    nets = ag.Nets([ag.task.image_classification.model_zoo.get_model('resnet18_v1'),
                    ag.task.image_classification.model_zoo.get_model('resnet34_v1'),
                    ag.task.image_classification.model_zoo.get_model('resnet50_v1'),
                    ag.task.image_classification.model_zoo.get_model('resnet101_v1'),
                    ag.task.image_classification.model_zoo.get_model('resnet152_v1')])
    logger.debug('nets:')
    logger.debug(nets)
    logger.debug('search space:')
    logger.debug(nets.search_space)
    for hparam in nets.search_space.get_hyperparameters():
        logger.debug(hparam.name)
    logger.debug('Finished.')


@pytest.mark.serial
def test_net_strs():
    logger.debug('Start testing nets')
    nets = ag.Nets(['resnet18_v1',
                    'resnet34_v1',
                    'resnet50_v1',
                    'resnet101_v1',
                    'resnet152_v1'])
    logger.debug('nets:')
    logger.debug(nets)
    logger.debug('search space:')
    logger.debug(nets.search_space)
    for hparam in nets.search_space.get_hyperparameters():
        logger.debug(hparam.name)
    logger.debug('Finished.')
