import pytest
import logging

from autogluon import TextClassification as task

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


@pytest.mark.serial
def test_text_classification_dataset():
    logger.debug('Start testing autogluon fit')
    dataset = task.Dataset(name='SST')
    logger.debug(dataset)
    logger.debug('Finished.')


@pytest.mark.serial
def test_text_classification_fit():
    dataset = task.Dataset(name='SST')
    predictor = task.fit(dataset)
    logger.debug('Top-1 val acc: %.3f' % predictor.results['best_reward'])
    logger.debug('The best configuration is:')
    logger.debug(predictor.results['best_config'])
    logger.debug('Finished.')


@pytest.mark.serial
def test_text_classification_predictor():
    dataset = task.Dataset(name='SST')
    predictor = task.fit(dataset)
    test_acc = predictor.evaluate(dataset)
    logger.debug('Top-1 test acc: %.3f' % test_acc)
    logger.debug('Finished.')
