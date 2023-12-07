# TODO: Standardize / unify this code with ag.save()
import json
import logging
import os
import tempfile

from ..utils import s3_utils

logger = logging.getLogger(__name__)


def save(path, obj, sanitize=True):
    if sanitize:
        obj = sanitize_object_to_primitives(obj=obj)
    is_s3_path = s3_utils.is_s3_url(path)
    if is_s3_path:
        import boto3

        data = json.dumps(obj).encode("UTF-8")

        bucket, key = s3_utils.s3_path_to_bucket_prefix(path)
        s3_client = boto3.client("s3")
        s3_client.put_object(Body=data, Bucket=bucket, Key=key)
    else:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(path, "w") as fp:
            json.dump(obj, fp, indent=2)


def sanitize_object_to_primitives(obj):
    if isinstance(obj, dict):
        obj_sanitized = dict()
        for key, val in obj.items():
            obj_sanitized[key] = sanitize_object_to_primitives(val)
    else:
        try:
            json.dumps(obj)
            obj_sanitized = obj
        except (TypeError, OverflowError):
            json.dumps(type(obj).__name__)
            obj_sanitized = type(obj).__name__
    return obj_sanitized
