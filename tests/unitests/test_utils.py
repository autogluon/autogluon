from __future__ import print_function

import pytest
import logging
import mxnet as mx

import autogluon as ag


@pytest.mark.serial
def test_dataset_santity_check():
    logging.info('Testing dataset santitycheck')
    a = mx.gluon.data.vision.CIFAR10(train=True)
    b = mx.gluon.data.vision.CIFAR10(train=False)
    ag.SanityCheck.check_dataset(a, b)
    logging.info('Finished.')

@pytest.mark.serial
def test_dataset_histogram_viz():
    logging.info('Testing dataset histogram viz')
    a = mx.gluon.data.vision.CIFAR10(train=True)
    b = mx.gluon.data.vision.CIFAR10(train=False)
    ag.Visualizer.visualize_dataset_label_histogram(a, b)
    logging.info('Finished.')
