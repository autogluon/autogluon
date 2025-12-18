from __future__ import annotations

import pytest


# -----------------------
# Helpers for fake HTTP
# -----------------------
class _FakeHTTPResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


# -----------------------
# Tests: YAML loader
# -----------------------
def test_load_https_with_fragment_selects_named_preset(monkeypatch):
    """
    https://.../presets.yaml#fast -> returns YAML['fast'] dict
    """
    # Import your module under test
    import autogluon.common.utils.presets_io as presets_io

    yaml_bytes = b"""
best_quality:
  auto_stack: true
  time_limit: 3600
fast:
  auto_stack: false
  time_limit: 120
"""

    # Patch urllib.request.urlopen used by _read_bytes_from_http
    def fake_urlopen(req, timeout=None):
        return _FakeHTTPResponse(yaml_bytes)

    monkeypatch.setattr(presets_io, "urlopen", fake_urlopen, raising=True)

    out = presets_io.load_preset_dict_from_location("https://autogluon.s3.us-west-2.amazonaws.com/presets.yaml#fast")
    assert out == {"auto_stack": False, "time_limit": 120}


def test_load_https_without_fragment_returns_top_level_dict(monkeypatch):
    """
    Old behavior: no fragment -> returns top-level dict.
    """
    import autogluon.common.utils.presets_io as presets_io

    yaml_bytes = b"""
auto_stack: true
time_limit: 3600
dynamic_stacking: auto
"""

    def fake_urlopen(req, timeout=None):
        return _FakeHTTPResponse(yaml_bytes)

    monkeypatch.setattr(presets_io, "urlopen", fake_urlopen, raising=True)

    out = presets_io.load_preset_dict_from_location("https://example.com/preset.yaml")
    assert out["auto_stack"] is True
    assert out["time_limit"] == 3600
    assert out["dynamic_stacking"] == "auto"


def test_load_fragment_missing_key_raises(monkeypatch):
    import autogluon.common.utils.presets_io as presets_io

    yaml_bytes = b"""
best_quality:
  auto_stack: true
"""

    def fake_urlopen(req, timeout=None):
        return _FakeHTTPResponse(yaml_bytes)

    monkeypatch.setattr(presets_io, "urlopen", fake_urlopen, raising=True)

    with pytest.raises(KeyError) as e:
        presets_io.load_preset_dict_from_location("https://example.com/presets.yaml#fast")
    assert "fast" in str(e.value)


# -----------------------
# Tests: s3:// read fallback to public HTTPS when no creds
# -----------------------
def test_load_s3_falls_back_to_public_https_when_no_credentials(monkeypatch):
    """
    If boto3 can't read due to missing creds, we try public HTTPS GET.
    No real S3/HTTP access: everything is mocked.
    """
    import autogluon.common.utils.presets_io as presets_io

    # This is what the public HTTPS fetch will return
    yaml_bytes = b"""
fast:
  auto_stack: false
  time_limit: 120
"""

    # Patch the public HTTP getter used by the S3 fallback path
    def fake_read_bytes_from_http(url: str, timeout_s: float = 30.0) -> bytes:
        # You can assert URL shape if you like:
        assert url.startswith("https://")
        return yaml_bytes

    monkeypatch.setattr(presets_io, "_read_bytes_from_http", fake_read_bytes_from_http, raising=True)

    # Now mock boto3 so get_object raises NoCredentialsError
    class _NoCredentialsError(Exception):
        pass

    class _FakeS3Client:
        def get_object(self, Bucket, Key):
            raise _NoCredentialsError("no creds")

    class _FakeBoto3:
        def client(self, name):
            assert name == "s3"
            return _FakeS3Client()

    # Monkeypatch boto3 import inside _read_bytes_from_s3 by setting module attribute
    monkeypatch.setattr(presets_io, "boto3", _FakeBoto3(), raising=False)

    # Monkeypatch exception types expected by your implementation.
    # If your code imports from botocore.exceptions, patch those symbols on presets_io.
    monkeypatch.setattr(presets_io, "NoCredentialsError", _NoCredentialsError, raising=False)
    monkeypatch.setattr(presets_io, "PartialCredentialsError", _NoCredentialsError, raising=False)

    out = presets_io.load_preset_dict_from_location("s3://autogluon/presets.yaml#fast")
    assert out == {"auto_stack": False, "time_limit": 120}


# -----------------------
# Tests: resolver behavior (original error when not a path)
# -----------------------
def test_unknown_preset_name_raises_original_valid_presets_error(monkeypatch):
    """
    If user passes a string that is NOT a known preset/alias AND does not look like a path/URL,
    we should raise the original error listing valid presets (and NOT attempt YAML loading).
    """
    import autogluon.common.utils.decorators as decorators

    preset_dict = {
        "best_quality": {"time_limit": 3600},
        "fast": {"time_limit": 120},
    }
    preset_alias = {"best": "best_quality"}

    # Guard: ensure loader would explode if called; it should NOT be called in this test.
    def bomb_loader(_):
        raise AssertionError("load_preset_dict_from_location should not be called for non-path preset names")

    # Patch wherever your resolver imports it from.
    # If your resolver does: from autogluon.common.utils.presets_io import load_preset_dict_from_location
    # then patch that symbol on the decorators module (or patch presets_io and ensure resolver uses it).
    monkeypatch.setattr(decorators, "load_preset_dict_from_location", bomb_loader, raising=False)

    with pytest.raises(ValueError) as e:
        decorators._resolve_preset_str("not_a_real_preset", preset_dict, preset_alias)

    msg = str(e.value)
    assert "Valid presets" in msg
    assert "best_quality" in msg
    assert "fast" in msg
    assert "best" in msg  # alias included (if your original error includes aliases)


def test_unknown_yaml_like_string_attempts_loading(monkeypatch):
    """
    If it looks like a YAML path/URL, the resolver should try to load it.
    """
    import autogluon.common.utils.decorators as decorators

    preset_dict = {"best_quality": {"time_limit": 3600}}
    preset_alias = {}

    def fake_loader(loc: str) -> dict:
        assert loc.endswith(".yaml")
        return {"time_limit": 999}

    monkeypatch.setattr(decorators, "load_preset_dict_from_location", fake_loader, raising=False)

    out = decorators._resolve_preset_str("my_preset.yaml", preset_dict, preset_alias)
    assert out == {"time_limit": 999}
