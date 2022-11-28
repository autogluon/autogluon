import boto3
import sagemaker

from autogluon.common.utils.s3_utils import is_s3_url, s3_path_to_bucket_prefix


def download_s3_file(bucket, prefix, path):
    s3 = boto3.client("s3")
    s3.download_file(bucket, prefix, path)


def is_s3_folder(path, session=None):
    """
    This function tries to determine if a s3 path is a folder.
    """
    assert is_s3_url(path)
    if session is None:
        session = sagemaker.session.Session()
    bucket, prefix = s3_path_to_bucket_prefix(path)
    contents = session.list_s3_files(bucket, prefix)
    if len(contents) > 1:
        return False
    # When the folder contains only 1 object, or the prefix is a file results in a len(contents) == 1
    # When the prefix is a file, the contents will match the prefix
    if contents[0] == prefix:
        return False
    return True
