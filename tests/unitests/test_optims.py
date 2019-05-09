from __future__ import print_function

import pytest
import logging

import autogluon as ag


@pytest.mark.serial
def test_optims():
    optims = ag.Optimizers([ag.Optimizers([ag.optims.get_optim('sgd'),
                                           ag.optims.get_optim('adam')])])
    logging.info(optims)
    pass
