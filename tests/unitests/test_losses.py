from __future__ import print_function

import pytest
import logging

import autogluon as ag

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


@pytest.mark.serial
def test_losses():
    logger.debug('Start testing losses')
    losses = ag.Losses([ag.task.image_classification.get_loss('L2Loss'),
                        ag.task.image_classification.get_loss('L1Loss'),
                        ag.task.image_classification.get_loss('SoftmaxCrossEntropyLoss')])
    print('losses:')
    print(losses)
    print('search space:')
    print(losses.search_space)
    for hparam in losses.search_space.get_hyperparameters():
        print(hparam.name)
    logger.debug('Finished.')


@pytest.mark.serial
def test_loss_strs():
    logger.debug('Start testing losses')
    losses = ag.Losses(['L2Loss',
                        'L1Loss',
                        'SoftmaxCrossEntropyLoss'])
    print('losses:')
    print(losses)
    print('search space:')
    print(losses.search_space)
    for hparam in losses.search_space.get_hyperparameters():
        print(hparam.name)
    logger.debug('Finished.')
