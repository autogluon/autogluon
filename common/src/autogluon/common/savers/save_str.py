import os
import logging

from ..utils import s3_utils

logger = logging.getLogger(__name__)


def save(path, data: str, verbose=True):
    """
    Saves the `data` value to a file.
    This function is compatible with local and s3 files.

    Parameters
    ----------
    path : str
        Path to the file to load the data from.
        Can be local or s3 path.
    data : str
        The string object to be saved.
    verbose : bool, default = True
        Whether to log that the file was saved.

    Examples
    --------
    >>> from autogluon.core.utils.loaders import load_str
    >>> from autogluon.core.utils.savers import save_str
    >>> data = 'the string value i want to save and load'
    >>> path = 'path/to/a/new/file'
    >>> save_str.save(path=path, data=data)
    >>> data_loaded = load_str.load(path=path)
    >>> assert data == data_loaded
    """

    is_s3_path = s3_utils.is_s3_url(path)
    if is_s3_path:
        import boto3
        bucket, key = s3_utils.s3_path_to_bucket_prefix(path)
        s3_client = boto3.client('s3')
        s3_client.put_object(Body=data, Bucket=bucket, Key=key)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(data)

    if verbose:
        logger.log(15, f'Saving {path} with contents "{data}"')
