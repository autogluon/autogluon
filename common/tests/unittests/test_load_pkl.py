import pickle
from unittest import mock

import pytest

from autogluon.common.loaders import load_pkl

_URL = "https://example.com/file.pkl"


def _mock_requests_get():
    """Patch requests.get so no real network call is made and a valid pickle is returned."""
    response = mock.MagicMock()
    response.content = pickle.dumps({"payload": 1})
    return mock.patch("requests.get", return_value=response)


def test_when_loading_pickle_from_url_without_opt_in_then_raises(monkeypatch):
    monkeypatch.delenv("AG_ALLOW_PICKLE_FROM_URL", raising=False)
    with pytest.raises(ValueError, match="Refusing to load a pickle file from a remote URL"):
        load_pkl.load(_URL)


def test_when_loading_pickle_from_url_with_trust_remote_then_loads(monkeypatch):
    monkeypatch.delenv("AG_ALLOW_PICKLE_FROM_URL", raising=False)
    with _mock_requests_get():
        assert load_pkl.load(_URL, trust_remote=True) == {"payload": 1}


def test_when_loading_pickle_from_url_with_env_var_then_loads(monkeypatch):
    monkeypatch.setenv("AG_ALLOW_PICKLE_FROM_URL", "True")
    with _mock_requests_get():
        assert load_pkl.load(_URL) == {"payload": 1}
