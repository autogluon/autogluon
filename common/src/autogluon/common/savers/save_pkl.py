# TODO: Standardize / unify this code with ag.save()
import logging
import os
import pickle
import tempfile

from ..utils import compression_utils, s3_utils

logger = logging.getLogger(__name__)

compression_fn_map = compression_utils.get_compression_map()


# TODO: object -> obj?
def save(path, object, format=None, verbose=True, **kwargs):
    compression_fn = kwargs.get("compression_fn", None)
    compression_fn_kwargs = kwargs.get("compression_fn_kwargs", None)

    if compression_fn in compression_fn_map:
        validated_path = compression_utils.get_validated_path(path, compression_fn)
    else:
        raise ValueError(
            f"compression_fn={compression_fn} is not a valid compression_fn. Valid values: {compression_fn_map.keys()}"
        )

    def pickle_fn(o, buffer):
        return pickle.dump(o, buffer, protocol=4)

    save_with_fn(
        validated_path,
        object,
        pickle_fn,
        format=format,
        verbose=verbose,
        compression_fn=compression_fn,
        compression_fn_kwargs=compression_fn_kwargs,
    )


def save_with_fn(path, object, pickle_fn, format=None, verbose=True, compression_fn=None, compression_fn_kwargs=None):
    if verbose:
        logger.log(15, "Saving " + str(path))
    if s3_utils.is_s3_url(path):
        format = "s3"
    if format == "s3":
        save_s3(path, object, pickle_fn, verbose=verbose)
    else:
        path_parent = os.path.dirname(path)
        if path_parent == "":
            path_parent = "."  # Allows saving to working directory root without crashing
        os.makedirs(path_parent, exist_ok=True)

        if compression_fn_kwargs is None:
            compression_fn_kwargs = {}

        with compression_fn_map[compression_fn]["open"](path, "wb", **compression_fn_kwargs) as fout:
            pickle_fn(object, fout)


def save_s3(path: str, obj, pickle_fn, verbose=True):
    import boto3

    if verbose:
        logger.info(f"save object to {path}")
    with tempfile.TemporaryFile() as f:
        pickle_fn(obj, f)
        f.flush()
        f.seek(0)

        bucket, key = s3_utils.s3_path_to_bucket_prefix(path)
        s3_client = boto3.client("s3")
        try:
            config = boto3.s3.transfer.TransferConfig()  # enable multipart uploading for files larger than 8MB
            s3_client.upload_fileobj(f, bucket, key, Config=config)
        except:
            logger.error("Failed to save object to s3")
            raise
