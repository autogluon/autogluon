from __future__ import print_function

import pytest
import logging
import mxnet as mx

import autogluon as ag


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


@pytest.mark.serial
def test_dataset_analyzer():
    logger.debug('Testing dataset santitycheck')
    a = mx.gluon.data.vision.CIFAR10(train=True)
    b = mx.gluon.data.vision.CIFAR10(train=False)
    ag.DataAnalyzer.check_dataset(a, b)
    logger.debug('Finished.')

@pytest.mark.serial
def test_dataset_histogram_viz():
    logger.debug('Testing dataset histogram viz')
    a = mx.gluon.data.vision.CIFAR10(train=True)
    b = mx.gluon.data.vision.CIFAR10(train=False)
    ag.Visualizer.visualize_dataset_label_histogram(a, b)
    logger.debug('Finished.')
