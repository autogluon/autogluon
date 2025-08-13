from __future__ import annotations

import json
import logging

from ..utils import s3_utils

logger = logging.getLogger(__name__)


def load(path: str, *, verbose=True) -> dict | list:
    if verbose:
        logger.log(15, "Loading: %s" % path)
    is_s3_path = s3_utils.is_s3_url(path)
    if is_s3_path:
        import boto3

        bucket, key = s3_utils.s3_path_to_bucket_prefix(path)
        s3_client = boto3.client("s3")
        result = s3_client.get_object(Bucket=bucket, Key=key)
        out = json.loads(result["Body"].read())
    else:
        with open(path, "r") as f:
            out = json.load(f)
    return out
