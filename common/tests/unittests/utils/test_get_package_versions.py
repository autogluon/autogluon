import types

import pytest

from autogluon.common.utils.utils import get_package_versions


class _FakeDist:
    def __init__(self, *, name=None, version="1.0", metadata=None, raise_on_metadata_get=False):
        self.name = name
        self.version = version
        self._metadata = metadata
        self._raise_on_metadata_get = raise_on_metadata_get

    @property
    def metadata(self):
        if self._metadata is None:
            return None
        if self._raise_on_metadata_get:
            # Simulate bizarre metadata implementations that raise unexpectedly
            class _Bad:
                def get(self, key):
                    raise RuntimeError("boom")
            return _Bad()
        return self._metadata


def test_get_package_versions_happy_path(monkeypatch):
    import importlib.metadata as im

    dists = [
        _FakeDist(metadata={"Name": "NumPy"}, version="2.0.0"),
        _FakeDist(metadata={"Name": "pandas"}, version="2.2.0"),
    ]
    monkeypatch.setattr(im, "distributions", lambda: iter(dists))

    versions, invalid = get_package_versions()
    assert versions == {"numpy": "2.0.0", "pandas": "2.2.0"}
    assert invalid == []


def test_get_package_versions_name_is_none_falls_back_to_dist_name(monkeypatch):
    import importlib.metadata as im

    dists = [
        _FakeDist(name="Scikit-Learn", metadata={"Name": None}, version="1.5.0"),
    ]
    monkeypatch.setattr(im, "distributions", lambda: iter(dists))

    versions, invalid = get_package_versions()
    assert versions == {"scikit-learn": "1.5.0"}
    assert invalid == []


def test_get_package_versions_missing_name_and_no_dist_name_is_skipped(monkeypatch):
    import importlib.metadata as im

    dists = [
        _FakeDist(name=None, metadata={"Name": None}, version="1.0"),
        _FakeDist(name=None, metadata=None, version="1.0"),
    ]
    monkeypatch.setattr(im, "distributions", lambda: iter(dists))

    versions, invalid = get_package_versions()
    assert versions == {}
    assert len(invalid) == 2


def test_get_package_versions_weird_metadata_does_not_crash(monkeypatch):
    import importlib.metadata as im

    dists = [
        _FakeDist(name="okpkg", metadata={"Name": "okpkg"}, version="0.1"),
        _FakeDist(name="fallbackpkg", metadata={"Name": "ignored"}, version="0.2", raise_on_metadata_get=True),
    ]
    monkeypatch.setattr(im, "distributions", lambda: iter(dists))

    versions, invalid = get_package_versions()
    # First uses metadata Name; second falls back to dist.name due to metadata.get raising.
    assert versions == {"okpkg": "0.1", "fallbackpkg": "0.2"}
    assert invalid == []


def test_get_package_versions_strict_raises(monkeypatch):
    import importlib.metadata as im

    class _ExplodingDist:
        @property
        def metadata(self):
            raise ValueError("bad dist")

    monkeypatch.setattr(im, "distributions", lambda: iter([_ExplodingDist()]))

    with pytest.raises(ValueError):
        get_package_versions(strict=True)
