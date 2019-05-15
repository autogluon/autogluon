from __future__ import print_function

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
    print('nets:')
    print(nets)
    print('search space:')
    print(nets.search_space)
    for hparam in nets.search_space.get_hyperparameters():
        print(hparam.name)
    logger.debug('Finished.')
