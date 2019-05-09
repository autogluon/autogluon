from __future__ import print_function

import pytest
import logging

import autogluon as ag


@pytest.mark.serial
def test_metrics():
    metrics = ag.Metrics([])
    logging.info(metrics)
    pass
