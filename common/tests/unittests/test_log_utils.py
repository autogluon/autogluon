import logging

import pytest
from numpy.testing import assert_almost_equal

from autogluon.common.utils.log_utils import convert_time_in_s_to_log_friendly, warn_if_mlflow_autologging_is_enabled


def test_convert_time_in_s_to_log_friendly():
    og_time = 1
    ne_time, time_unit = convert_time_in_s_to_log_friendly(og_time)
    assert_almost_equal(og_time, ne_time)
    assert time_unit == "s"

    og_time = 10
    ne_time, time_unit = convert_time_in_s_to_log_friendly(og_time)
    assert_almost_equal(og_time, ne_time)
    assert time_unit == "s"

    og_time = 10000
    ne_time, time_unit = convert_time_in_s_to_log_friendly(og_time)
    assert_almost_equal(og_time, ne_time)
    assert time_unit == "s"

    og_time = 0.1
    ne_time, time_unit = convert_time_in_s_to_log_friendly(og_time)
    assert_almost_equal(og_time, ne_time)
    assert time_unit == "s"

    og_time = 0.01
    ne_time, time_unit = convert_time_in_s_to_log_friendly(og_time)
    assert_almost_equal(og_time, ne_time)
    assert time_unit == "s"

    og_time = 0.0099
    ne_time, time_unit = convert_time_in_s_to_log_friendly(og_time)
    assert_almost_equal(9.9, ne_time)
    assert time_unit == "ms"

    og_time = 0.00001
    ne_time, time_unit = convert_time_in_s_to_log_friendly(og_time)
    assert_almost_equal(0.01, ne_time)
    assert time_unit == "ms"
    ne_time, time_unit = convert_time_in_s_to_log_friendly(og_time, min_value=1)
    assert_almost_equal(10, ne_time)
    assert time_unit == "μs"

    og_time = 0.0000099
    ne_time, time_unit = convert_time_in_s_to_log_friendly(og_time)
    assert_almost_equal(9.9, ne_time)
    assert time_unit == "μs"

    og_time = 0.00000001
    ne_time, time_unit = convert_time_in_s_to_log_friendly(og_time)
    assert_almost_equal(0.01, ne_time)
    assert time_unit == "μs"

    og_time = 0.0000000099
    ne_time, time_unit = convert_time_in_s_to_log_friendly(og_time)
    assert_almost_equal(9.9, ne_time)
    assert time_unit == "ns"

    og_time = 0.00000000001
    ne_time, time_unit = convert_time_in_s_to_log_friendly(og_time)
    assert_almost_equal(0.01, ne_time)
    assert time_unit == "ns"

    og_time = 0.0000000000099
    ne_time, time_unit = convert_time_in_s_to_log_friendly(og_time)
    assert_almost_equal(0.0099, ne_time)
    assert time_unit == "ns"


def test_when_mlflow_autolog_is_disabled_then_no_warning_is_logged(caplog):
    logger = logging.getLogger("autogluon")
    logger.propagate = True
    try:
        import mlflow

        pytest.fail("mlflow shouldn't be installed in the test env")
    except ImportError:
        with caplog.at_level(logging.DEBUG, logger="autogluon"):
            warn_if_mlflow_autologging_is_enabled()
        assert len(caplog.records) == 0
