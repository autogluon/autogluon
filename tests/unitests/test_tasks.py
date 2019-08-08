from __future__ import print_function

import pytest
import logging

import autogluon as ag

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


@pytest.mark.serial
def test_image_classification():
    logger.debug('Start testing image classification')
    for name in ag.task.image_classification.models:
        model = ag.task.image_classification.get_model(name)
        print(model.name)
    logger.debug('Finished.')


@pytest.mark.serial
def test_text_classification():
    logger.debug('Start testing text classification')
    for name in ag.task.text_classification.models:
        model = ag.task.text_classification.get_model(name)
        print(model.name)
    logger.debug('Finished.')
