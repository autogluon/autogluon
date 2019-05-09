from __future__ import print_function

import pytest
import logging

import autogluon as ag


@pytest.mark.serial
def test_image_classification():
    for name in ag.task.image_classification.models:
        model = ag.task.image_classification.get_model(name)
        logging.info(model)
    pass
