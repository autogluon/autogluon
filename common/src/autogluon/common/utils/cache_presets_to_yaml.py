from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse


def _is_s3_uri(path: str) -> bool:
    return urlparse(path).scheme == "s3"


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    p = urlparse(uri)
    if p.scheme != "s3":
        raise ValueError(f"Not an s3 uri: {uri!r}")
    bucket = p.netloc
    key = p.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Invalid s3 uri (expected s3://bucket/key): {uri!r}")
    return bucket, key


def _s3_put_text(
    uri: str,
    text: str,
    *,
    content_type: str = "text/yaml; charset=utf-8",
    upload_as_public: bool = False,
) -> None:
    import boto3

    bucket, key = _parse_s3_uri(uri)

    put_kwargs = dict(
        Bucket=bucket,
        Key=key,
        Body=text.encode("utf-8"),
        ContentType=content_type,
    )

    if upload_as_public:
        put_kwargs["ACL"] = "public-read"

    boto3.client("s3").put_object(**put_kwargs)


def presets_to_yaml_files(
    presets: Mapping[str, Mapping[str, Any]] | Mapping[str, Any],
    output: str | Path,
    *,
    multi: bool | None = None,
    per_preset_files: bool = False,
    overwrite: bool = False,
    sort_keys: bool = False,
    upload_as_public: bool = False,  # NEW
) -> list[str]:
    """
    Export AutoGluon-style presets to YAML.

    Supports:
      - Local file output (single file) or local directory output (per preset files)
      - S3 output:
          * single file:  s3://bucket/path/presets.yaml
          * per preset:   s3://bucket/path/presets/   (prefix; each becomes <name>.yaml)

    Notes
    -----
    - overwrite=False cannot be reliably enforced on S3 without an extra HEAD call; this implementation
      will enforce overwrite for local paths, and will *best-effort* enforce it for S3 via head_object.
    - Returns list of destinations written (local paths or s3 uris as strings).
    """
    output_str = str(output)

    def _infer_is_multi(obj: Mapping[str, Any]) -> bool:
        if not obj:
            return False
        return all(isinstance(v, Mapping) for v in obj.values())

    if multi is None:
        multi = _infer_is_multi(presets)  # type: ignore[arg-type]

    if multi:
        if not isinstance(presets, Mapping) or (presets and not all(isinstance(v, Mapping) for v in presets.values())):
            raise TypeError("multi=True expects named presets like {'best_quality': {...}, 'fast': {...}}")
        named_presets: dict[str, dict[str, Any]] = {str(k): dict(v) for k, v in presets.items()}  # type: ignore[assignment]
    else:
        single_preset: dict[str, Any] = dict(presets)  # type: ignore[arg-type]

    def _dump_yaml_text(obj: Any) -> str:
        import yaml

        return yaml.safe_dump(
            obj,
            default_flow_style=False,
            sort_keys=sort_keys,
            allow_unicode=True,
        )

    def _local_write_text(path: Path, text: str) -> None:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _s3_exists(uri: str) -> bool:
        # best-effort exists check
        import boto3
        from botocore.exceptions import ClientError

        bucket, key = _parse_s3_uri(uri)
        try:
            boto3.client("s3").head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            # If access is denied, we can't determine existence; treat as "unknown"
            if code in ("403", "AccessDenied"):
                return False
            raise

    written: list[str] = []

    # ---- per-preset output ----
    if per_preset_files:
        if not multi:
            raise ValueError("per_preset_files=True requires multi=True / named presets input.")

        if _is_s3_uri(output_str):
            # Interpret output as a prefix (ensure it ends with '/')
            prefix_uri = output_str if output_str.endswith("/") else (output_str + "/")
            for name, preset in named_presets.items():
                uri = f"{prefix_uri}{name}.yaml"
                if not overwrite and _s3_exists(uri):
                    raise FileExistsError(f"Refusing to overwrite existing S3 object: {uri}")
                _s3_put_text(uri, _dump_yaml_text(preset), upload_as_public=upload_as_public)
                written.append(uri)
            return written
        else:
            out_dir = Path(output_str)
            out_dir.mkdir(parents=True, exist_ok=True)
            for name, preset in named_presets.items():
                path = out_dir / f"{name}.yaml"
                _local_write_text(path, _dump_yaml_text(preset))
                written.append(str(path))
            return written

    # ---- single-file output ----
    if _is_s3_uri(output_str):
        # output must be a concrete key ending with .yml/.yaml
        if not output_str.lower().endswith((".yaml", ".yml")):
            raise ValueError(f"S3 output must be a .yaml/.yml object key: {output_str!r}")

        if not overwrite and _s3_exists(output_str):
            raise FileExistsError(f"Refusing to overwrite existing S3 object: {output_str}")

        obj = named_presets if multi else single_preset
        _s3_put_text(output_str, _dump_yaml_text(obj), upload_as_public=upload_as_public)
        written.append(output_str)
        return written

    # local single file
    out_path = Path(output_str)
    if out_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"output must be a .yaml/.yml file when per_preset_files=False: {out_path}")
    obj = named_presets if multi else single_preset
    _local_write_text(out_path, _dump_yaml_text(obj))
    written.append(str(out_path))
    return written
