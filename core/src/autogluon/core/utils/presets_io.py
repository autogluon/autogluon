from __future__ import annotations

import os
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

import yaml


def _read_bytes_from_s3(uri: str) -> bytes:
    import boto3

    p = urlparse(uri)
    bucket = p.netloc
    key = p.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Invalid s3 uri: {uri}")

    obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def _read_bytes_from_http(uri: str, *, timeout_s: float = 30.0) -> bytes:
    # Basic urllib implementation; no extra deps.
    # Add a UA to avoid some CDNs rejecting default python-urllib.
    req = Request(uri, headers={"User-Agent": "AutoGluon/PresetsLoader"})
    with urlopen(req, timeout=timeout_s) as r:
        return r.read()


def load_preset_dict_from_location(location: str) -> dict:
    """
    Load a preset dict from YAML at:
      - local path
      - s3://bucket/key
      - http(s)://...

    Fragment support:
      - .../presets.yaml#fast -> returns YAML['fast'] (must be dict)
      - no fragment -> returns the YAML top-level dict (backwards compatible)
    """
    p = urlparse(location)
    fragment = p.fragment or None

    # Strip fragment for retrieval
    p_no_frag = p._replace(fragment="")
    location_no_frag = urlunparse(p_no_frag)

    if p.scheme in ("", "file"):
        path = location_no_frag if p.scheme == "" else p_no_frag.path
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            data = f.read()
    elif p.scheme == "s3":
        data = _read_bytes_from_s3(location_no_frag)
    elif p.scheme in ("http", "https"):
        data = _read_bytes_from_http(location_no_frag)
    else:
        raise ValueError(f"Unsupported preset URI scheme {p.scheme!r} in {location!r}")

    loaded = yaml.safe_load(data)
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Preset YAML must be a dict at top-level, got {type(loaded)} from {location!r}")

    if fragment is not None:
        if fragment not in loaded:
            raise KeyError(
                f"Preset fragment {fragment!r} not found in {location_no_frag!r}. "
                f"Available keys: {list(loaded.keys())}"
            )
        selected = loaded[fragment] or {}
        if not isinstance(selected, dict):
            raise TypeError(
                f"Preset {fragment!r} in {location_no_frag!r} must be a dict, got {type(selected)}"
            )
        return selected

    return loaded
