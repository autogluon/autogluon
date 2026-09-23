from __future__ import annotations

import multiprocessing

import pytest

from autogluon.common.utils.multiprocessing_utils import execute_multiprocessing


def _double(x):
    return x * 2


def test_execute_multiprocessing_falls_back_when_method_unavailable():
    result = execute_multiprocessing(
        workers_count=2,
        transformer=_double,
        chunks=[1, 2, 3, 4],
        multiprocessing_method="forkserver",
    )
    assert sorted(result) == [2, 4, 6, 8]


def test_execute_multiprocessing_unavailable_method_uses_spawn(monkeypatch):
    """The fallback substitutes 'spawn' when the requested method is unavailable."""
    import autogluon.common.utils.multiprocessing_utils as mp_utils

    monkeypatch.setattr(multiprocessing, "get_all_start_methods", lambda: ["spawn"])

    seen = {}
    real_get_context = multiprocessing.get_context

    def spy_get_context(method):
        seen["method"] = method
        return real_get_context(method)

    monkeypatch.setattr(mp_utils.multiprocessing, "get_context", spy_get_context)

    result = mp_utils.execute_multiprocessing(
        workers_count=1,
        transformer=_double,
        chunks=[5],
        multiprocessing_method="forkserver",
    )
    assert result == [10]
    assert seen["method"] == "spawn"


@pytest.mark.parametrize("method", ["frok", "Forkserver", "threads"])
def test_execute_multiprocessing_unknown_method_raises(monkeypatch, method):
    """A name that is not a start method raises instead of falling back, also where only 'spawn' is available."""
    monkeypatch.setattr(multiprocessing, "get_all_start_methods", lambda: ["spawn"])
    with pytest.raises(ValueError, match="cannot find context"):
        execute_multiprocessing(workers_count=1, transformer=_double, chunks=[1], multiprocessing_method=method)


@pytest.mark.skipif(
    "forkserver" not in multiprocessing.get_all_start_methods(),
    reason="requires a platform where 'forkserver' is natively available",
)
def test_execute_multiprocessing_uses_requested_method_when_available(monkeypatch):
    """When the requested method IS available, it must be used as-is (no unwanted fallback)."""
    import autogluon.common.utils.multiprocessing_utils as mp_utils

    seen = {}
    real_get_context = multiprocessing.get_context

    def spy_get_context(method):
        seen["method"] = method
        return real_get_context(method)

    monkeypatch.setattr(mp_utils.multiprocessing, "get_context", spy_get_context)

    result = mp_utils.execute_multiprocessing(
        workers_count=1,
        transformer=_double,
        chunks=[3],
        multiprocessing_method="forkserver",
    )
    assert result == [6]
    assert seen["method"] == "forkserver"
