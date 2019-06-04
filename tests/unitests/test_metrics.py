from __future__ import print_function

import pytest
import logging

import autogluon as ag

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


@pytest.mark.serial
def test_metrics():
    logger.debug('Start testing metrics')
    metrics = ag.Metrics([ag.task.image_classification.get_metric('Accuracy'),
                          ag.task.image_classification.get_metric('TopKAccuracy'),
                          ag.task.image_classification.get_metric('F1')])
    print('metrics:')
    print(metrics)
    print('search space:')
    print(metrics.search_space)
    for hparam in metrics.search_space.get_hyperparameters():
        print(hparam.name)
    logger.debug('Finished.')


@pytest.mark.serial
def test_metric_strs():
    logger.debug('Start testing metrics')
    metrics = ag.Metrics(['Accuracy',
                          'TopKAccuracy',
                          'F1'])
    print('metrics:')
    print(metrics)
    print('search space:')
    print(metrics.search_space)
    for hparam in metrics.search_space.get_hyperparameters():
        print(hparam.name)
    logger.debug('Finished.')
