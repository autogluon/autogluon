import logging

from ..utils import s3_utils

logger = logging.getLogger(__name__)


def load(path: str) -> str:
    """
    Loads the `data` value from a file saved via `savers.save_str.save(path=path, data=data)`.
    This function is compatible with local and s3 files.

    Parameters
    ----------
    path : str
        Path to the file to load the data from.
        Can be local or s3 path.

    Returns
    -------
    data : str
        The string object that is contained in the loaded file.

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
        s3_object = s3_client.get_object(Bucket=bucket, Key=key)
        data = s3_object['Body'].read().decode("utf-8")
    else:
        with open(path, "r") as f:
            data = f.read()
    return data
