from __future__ import print_function

import pytest
import logging

import autogluon as ag

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


@pytest.mark.serial
def test_losses():
    losses = ag.Losses([])
    logger.debug(losses)
