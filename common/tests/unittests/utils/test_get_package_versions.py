import pytest

from autogluon.common.utils.utils import (
    _get_package_versions_cached,
    _name_and_version_from_metadata_header,
    get_package_versions,
)


@pytest.fixture(autouse=True)
def _fresh_package_versions():
    """Each test enumerates its own fake distributions, so the per-process cache is emptied around it."""
    _get_package_versions_cached.cache_clear()
    yield
    _get_package_versions_cached.cache_clear()


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


def test_get_package_versions_is_computed_once_per_process(monkeypatch):
    import importlib.metadata as im

    calls = []

    def distributions():
        calls.append(1)
        return iter([_FakeDist(metadata={"Name": "NumPy"}, version="2.0.0")])

    monkeypatch.setattr(im, "distributions", distributions)
    first, first_invalid = get_package_versions()
    second, second_invalid = get_package_versions()
    assert first == second == {"numpy": "2.0.0"} and first_invalid == second_invalid == []
    assert len(calls) == 1
    # callers get their own containers: mutating one result must not leak into the next
    first["numpy"] = "changed"
    first_invalid.append("x")
    third, third_invalid = get_package_versions()
    assert third == {"numpy": "2.0.0"} and third_invalid == []


class _DiskDist:
    """A distribution whose metadata lives on disk, like importlib's `PathDistribution`."""

    def __init__(self, path, *, name="Fallback", version="9.9"):
        self._path = path
        self.name = name
        self.version = version
        self.metadata = {"Name": name}


def test_metadata_header_read(tmp_path):
    dist_info = tmp_path / "some_pkg-1.2.3.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: Some_Pkg\nVersion: 1.2.3\nSummary: x\n\n"
        "Name: not-the-name\nVersion: 0.0\nlong description\n",
        encoding="utf-8",
    )
    assert _name_and_version_from_metadata_header(_DiskDist(dist_info)) == ("Some_Pkg", "1.2.3")

    egg_info = tmp_path / "legacy.egg-info"
    egg_info.mkdir()
    (egg_info / "PKG-INFO").write_text("Metadata-Version: 1.0\nName: legacy\nVersion: 0.1\n", encoding="utf-8")
    assert _name_and_version_from_metadata_header(_DiskDist(egg_info)) == ("legacy", "0.1")

    empty = tmp_path / "empty.dist-info"
    empty.mkdir()
    assert _name_and_version_from_metadata_header(_DiskDist(empty)) == (None, None)
    (empty / "METADATA").write_text("Version: 1.0\n\n", encoding="utf-8")  # no Name: fall back
    assert _name_and_version_from_metadata_header(_DiskDist(empty)) == (None, None)
    assert _name_and_version_from_metadata_header(_FakeDist(metadata={"Name": "x"})) == (None, None)


def test_get_package_versions_prefers_the_header_and_falls_back_to_metadata(tmp_path, monkeypatch):
    import importlib.metadata as im

    dist_info = tmp_path / "some_pkg-1.2.3.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Name: Some_Pkg\nVersion: 1.2.3\n\nbody\n", encoding="utf-8")
    on_disk = _DiskDist(dist_info, name="ignored", version="0")
    no_file = _DiskDist(tmp_path / "missing.dist-info", name="FromMetadata", version="4.5")
    monkeypatch.setattr(im, "distributions", lambda: iter([on_disk, no_file]))
    versions, invalid = get_package_versions()
    assert versions == {"some_pkg": "1.2.3", "frommetadata": "4.5"}
    assert invalid == []


def test_get_package_versions_matches_the_full_metadata_parse_on_this_environment():
    """The header read must agree with `Distribution.metadata` for every installed distribution."""
    import importlib.metadata as im

    expected = {}
    for dist in im.distributions():
        name = dist.metadata.get("Name") if dist.metadata is not None else None
        name = name or getattr(dist, "name", None)
        if name:
            expected[str(name).lower()] = str(dist.version) if dist.version is not None else "unknown"
    versions, _ = get_package_versions()
    assert versions == expected
