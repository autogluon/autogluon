from __future__ import print_function

import pytest
import logging

import autogluon as ag


@pytest.mark.serial
def test_list_space():
    list_space = ag.space.List('listspace', ['0',
                                             '1',
                                             '2'])
    logging.info(list_space)
    pass


@pytest.mark.serial
def test_linear_space():
    linear_space = ag.space.Linear('linspace', 0, 10)
    logging.info(linear_space)
    pass


@pytest.mark.serial
def test_log_space():
    log_space = ag.space.Log('logspace', 10**-10, 10**-1)
    logging.info(log_space)
    pass
