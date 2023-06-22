import logging
import os
import pathlib
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


def list_bucket_s3(bucket):
    import boto3

    logger.log(15, "Listing s3 bucket: " + str(bucket))

    s3bucket = boto3.resource("s3")
    my_bucket = s3bucket.Bucket(bucket)
    files = []
    for object in my_bucket.objects.all():
        files.append(object.key)
        logger.log(15, str(object.key))
    return files


def download(input_bucket, input_prefix, local_path):
    import boto3

    directory = os.path.dirname(local_path)
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

    s3 = boto3.resource("s3")
    s3.Bucket(input_bucket).download_file(input_prefix, local_path)


# TODO: Expand to support arbitrary list of candidate file paths
# TODO: add local file path support as a more general function
# TODO: Consider renaming and deprecating old name
# TODO: consider using a single parameter supporting wildcards or regex - this will solve all possible use cases for filtering
#   list_bucket_prefix_suffix_contains_s3(..., exclude=['**/*.bak', '**/data/*_excl.csv'])
# TODO: Add unit tests for non-boto3 logic
def list_bucket_prefix_suffix_contains_s3(
    bucket: str,
    prefix: str,
    suffix: Optional[Union[str, List[str]]] = None,
    exclude_suffix: Optional[Union[str, List[str]]] = None,
    contains: Optional[Union[str, List[str]]] = None,
    exclude_contains: Optional[Union[str, List[str]]] = None,
) -> List[str]:
    """
    Returns a list of file paths within an S3 bucket that satisfies the constraints.

    Parameters
    ----------
    bucket : str
        The S3 bucket to list files from.
        You must have read permissions to the S3 bucket and its files for this function to work.
    prefix : str
        The string prefix to search for files within the S3 bucket. Any file outside of this prefix will not be considered.
        For example, if `bucket='autogluon'` and `prefix='datasets/'`,
        only files starting under `s3://autogluon/datasets/` will be considered.
        To check all files in the bucket, specify `prefix=''` (empty string)
    suffix : str or List[str], default = None
        If specified, filters files to ensure their paths end with the specified suffix (if str)
        or at least one element of `suffix` (if list) in the post-prefix string path.
    exclude_suffix : str or List[str], default = None
        If specified, filters files to ensure their paths do not end with any element in `exclude_suffix`.
    contains : str or List[str], default = None
        If specified, will filter any result that doesn't contain `contains` (if str)
        or at least one element of `contains` (if list) in the post-prefix string path.
    exclude_contains : str or List[str], default = None
        If specified, filters files to ensure their paths do not contain any element in `exclude_contains`.

    Returns
    -------
    Returns a list of file paths within an S3 bucket that satisfies the constraints.

    """
    import boto3

    if exclude_suffix is None:
        exclude_suffix = []
    if exclude_contains is None:
        exclude_contains = []
    if suffix is not None and not isinstance(suffix, list):
        suffix = [suffix]
    if exclude_suffix is not None and not isinstance(exclude_suffix, list):
        exclude_suffix = [exclude_suffix]
    if contains is not None and not isinstance(contains, list):
        contains = [contains]
    if exclude_contains is not None and not isinstance(exclude_contains, list):
        exclude_contains = [exclude_contains]

    s3 = boto3.resource("s3")
    my_bucket = s3.Bucket(bucket)

    files = []
    for object_summary in my_bucket.objects.filter(Prefix=prefix):
        suffix_full = object_summary.key.split(prefix, 1)[1] if len(prefix) > 0 else object_summary.key
        is_banned = False
        for banned_s in exclude_suffix:
            if suffix_full.endswith(banned_s):
                is_banned = True
                break
        if is_banned:
            continue
        for banned_c in exclude_contains:
            if banned_c in suffix_full:
                is_banned = True
                break
        if is_banned:
            continue
        if suffix is not None:
            has_valid_suffix = False
            for s in suffix:
                if suffix_full.endswith(s):
                    has_valid_suffix = True
                    break
            if not has_valid_suffix:
                continue
        if contains is None:
            files.append(object_summary.key)
        else:
            for c in contains:
                if c in suffix_full:
                    files.append(object_summary.key)
                    break
    return files
