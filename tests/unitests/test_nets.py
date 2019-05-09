from __future__ import print_function

import pytest
import logging

import autogluon as ag


@pytest.mark.serial
def test_nets():
    nets = ag.Nets([ag.task.model_zoo.get_model('resnet18_v1'),
                    ag.task.model_zoo.get_model('resnet34_v1'),
                    ag.task.model_zoo.get_model('resnet50_v1'),
                    ag.task.model_zoo.get_model('resnet101_v1'),
                    ag.task.model_zoo.get_model('resnet152_v1')])
    logging.info(nets)
    pass
