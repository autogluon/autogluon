import logging
import os
import pathlib
import shutil

from typing import Dict, List, Tuple, Optional, Union

from ..loaders.load_s3 import list_bucket_prefix_suffix_contains_s3


logger = logging.getLogger(__name__)


def is_s3_url(path):
    if (path[:2] == 's3') and ('://' in path[:6]):
        return True
    return False


def s3_path_to_bucket_prefix(s3_path):
    s3_path_cleaned = s3_path.split('://', 1)[1]
    bucket, prefix = s3_path_cleaned.split('/', 1)

    return bucket, prefix


def s3_bucket_prefix_to_path(bucket, prefix, version='s3'):
    return version + '://' + bucket + '/' + prefix


def delete_s3_prefix(bucket, prefix):
    import boto3
    s3 = boto3.resource('s3')
    objects_to_delete = s3.meta.client.list_objects(Bucket=bucket, Prefix=prefix)

    delete_keys = {'Objects': []}
    delete_keys['Objects'] = [{'Key': k} for k in [obj['Key'] for obj in objects_to_delete.get('Contents', [])]]

    if len(delete_keys['Objects']) != 0:
        s3.meta.client.delete_objects(Bucket=bucket, Delete=delete_keys)
        

def upload_file(file_name: str, bucket: str, prefix: Optional[str] = None):
    """
    Upload a file to a S3 bucket

    Parameters
    ----------
    file_name: str,
        File to upload
    bucket: str,
        Bucket to upload to
    prefix: Optional[str], default = None
        S3 prefix. If not specified then will upload to the root of the bucket
    """
    import boto3
    from botocore.exceptions import ClientError

    object_name = os.path.basename(file_name)
    if len(prefix) == 0:
        prefix = None
    if prefix is not None:
        object_name = prefix + "/" + object_name

    # Upload the file
    s3_client = boto3.client("s3")
    s3_client.upload_file(file_name, bucket, object_name)
    
    
def upload_s3_folder(
    bucket: str,
    prefix: str,
    folder_to_upload: str,
    dry_run: bool = False,
    verbose: bool = True
):
    """
    Upload a folder to a S3 bucket and maintain its inner structure
    For example, assuming bucket = bar and prefix = foo, and folder_to_upload looks like this:
    .
    └── folder_to_upload/
        └── test.txt/
            └── temp/
                └── test2.txt
    After uploading to s3, the bucket would look like this:
    .
    └── bar/
        └── foo/
            └── test.txt/
                └── temp/
                    └── test2.txt

    Parameters
    ----------
    bucket: str
        The name of the bucket
    prefix: str
        The prefix to upload to
        To upload to the root of the bucket, specify `prefix=''` (empty string)
    folder_to_upload: str
        The local folder to upload to s3
    dry_run: bool, default = False
        Whether to perform uploading
        If True, will isntead log every file that will be uploaded and the s3 path to be uploaded to
    verbose: bool, default = True
        Whether to log detailed loggings
    """
    if prefix.endswith("/"):
        prefix = prefix[:-1]
    files_to_upload = _get_local_objs_to_upload_and_s3_prefix(folder_to_upload=folder_to_upload)
    if verbose:
        logger.log(20, f"Will upload {len(files_to_upload)} objects from {folder_to_upload} to s3://{bucket}/{prefix}")
    for file_local_path, file_s3_path in files_to_upload:
        file_prefix = prefix + "/" + file_s3_path if len(prefix) > 0 else file_s3_path
        if dry_run:
            logger.log(20, f"Will upload {file_local_path} to s3://{bucket}/{file_prefix}")
        else:
            file_prefix = os.path.dirname(file_prefix)
            upload_file(
                file_name=file_local_path,
                bucket=bucket,
                prefix=file_prefix if len(file_prefix) > 0 else None
            )


# TODO: Add remaining arguments from list_bucket_prefix_suffix_contains_s3
def download_s3_folder(
    *,
    bucket: str,
    prefix: str,
    local_path: str,
    suffix: Optional[Union[str, List[str]]] = None,
    error_if_exists: bool = True,
    delete_if_exists: bool = False,
    dry_run: bool = False,
    verbose: bool = True
):
    """
    This util function downloads a s3 folder and maintain its structure.
    For example, assuming bucket = bar and prefix = foo, and the bucket structure looks like this
        .
        └── bar/
            ├── test.txt
            └── foo/
                ├── test2.txt
                └── temp/
                    └── test3.txt
    This util will download foo to `local_path` and maintain the structure:
        .
        └── local_path/
            └── test2.txt/
                └── temp/
                    └── test3.txt
                    
    Parameters
    ----------
    bucket: str
        The name of the bucket
    prefix: str
        The prefix of the folder to be downloaded
        To check all files in the bucket, specify `prefix=''` (empty string)
    local_path: str
        The local path to download the object/folder into
    suffix : str or List[str], default = None
        If specified, filters files to ensure their paths end with the specified suffix (if str)
        or at least one element of `suffix` (if list) in the post-prefix string path.
    error_if_exists: bool, default = True
        Whether to raise an error if the root folder exists already
    delete_if_exists: bool, default = False
        Whether to delete the local root folder and all contents within if the root folder exists already
        If `error_if_exists=True`, deletion will not occur.
    dry_run: bool, default = False
        Whether to perform the directory creation and file downloading.
        If True, will isntead log every file that will be downloaded and every directory that will be created
    verbose: bool, default = True
        Whether to log detailed loggings
    """
    s3_to_local_tuple_list = get_s3_to_local_tuple_list_from_s3_folder(bucket=bucket,
                                                                       prefix=prefix,
                                                                       local_path=local_path,
                                                                       suffix=suffix)
    if verbose:
        logger.log(20, f"Will download {len(s3_to_local_tuple_list)} objects from s3://{bucket}/{prefix} to {local_path}")
    if os.path.isdir(local_path) and not dry_run:
        if error_if_exists:
            raise ValueError(f"Directory {local_path} already exsists. Please pass in a different `local_path` or set `error_if_exsits` to `False`")
        if delete_if_exists:
            logger.warning(f"Will delete {local_path} and all its content within because this folder already exists and `delete_if_exists` = `True`")
            shutil.rmtree(local_path)
    download_s3_files(s3_to_local_tuple_list=s3_to_local_tuple_list, dry_run=dry_run)


def get_s3_to_local_tuple_list_from_s3_folder(*,
                                              bucket: str,
                                              prefix: str,
                                              local_path: str,
                                              **kwargs) -> List[Tuple[str, str]]:
    """
    Given a s3 bucket and prefix, as well as a target local prefix, return a list of tuples of (s3_path, local_path)
    indicating the origin to target file path when downloading from S3 for each file.

    This output can be passed directly into `download_s3_files(s3_to_local_tuple_list)` to download the s3 files.
    """
    if len(prefix) > 0:
        assert prefix.endswith("/"), "Please provide a prefix to a folder and end it with '/'"
    objs = list_bucket_prefix_suffix_contains_s3(bucket=bucket, prefix=prefix, **kwargs)
    s3_to_local_tuple_list = get_s3_to_local_tuple_list(s3_bucket=bucket,
                                                        s3_prefix=prefix,
                                                        local_prefix=local_path,
                                                        s3_prefixes=objs)
    return s3_to_local_tuple_list


# TODO: Add unit tests
def get_s3_to_local_tuple_list(*,
                               s3_bucket: str,
                               s3_prefix: str,
                               local_prefix: str,
                               s3_prefixes: List[str]) -> List[Tuple[str, str]]:
    """
    Given a list of s3 objects and a target local prefix, return a list of tuples of (s3_path, local_path)
    indicating the origin to target file path when downloading from s3.

    This output can be passed directly into `download_s3_files(s3_to_local_tuple_list)` to download the s3 files.
    """
    local_paths = _get_local_path_to_download_objs(
        s3_objs=s3_prefixes,
        prefix=s3_prefix,
        local_path=local_prefix
    )
    s3_to_local_tuple_list = []
    for s3_prefix, local_path in zip(s3_prefixes, local_paths):
        s3_path = s3_bucket_prefix_to_path(bucket=s3_bucket, prefix=s3_prefix)
        s3_to_local_tuple_list.append((s3_path, local_path))
    return s3_to_local_tuple_list


def download_s3_file(*, s3_path: str, local_path: str, mkdir: bool = True, dry_run: bool = False):
    """
    Download a file from s3 to local.
    """
    s3_bucket, s3_prefix = s3_path_to_bucket_prefix(s3_path=s3_path)
    if dry_run:
        logger.log(20, f'Dry Run: Would download S3 file "{s3_path}" to "{local_path}"')
    else:
        import boto3
        if mkdir:
            directory = os.path.dirname(local_path)
            if directory not in ['', '.']:
                pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        s3 = boto3.resource('s3')
        s3.Bucket(s3_bucket).download_file(s3_prefix, local_path)


def download_s3_files(*, s3_to_local_tuple_list: List[Tuple[str, str]], dry_run: bool = False):
    """
    For (s3_path, local_path) in `s3_to_local_tuple_list`, call `download_s3_file`.
    """
    for s3_path, _ in s3_to_local_tuple_list:
        assert is_s3_url(path=s3_path), f'S3 path is not a valid S3 URL: "{s3_path}"'
    for s3_path, local_path in s3_to_local_tuple_list:
        download_s3_file(s3_path=s3_path, local_path=local_path, mkdir=True, dry_run=dry_run)


def _get_local_objs_to_upload_and_s3_prefix(folder_to_upload: str) -> List[Tuple[str, str]]:
    """
    Get paths to all the objects within a folder and it's subfolder and their relative path to maintain the folder structure
    
    Parameters
    ----------
    folder_to_upload: str
        The local folder to upload to s3
        
    Returns
    --------
    List[Tuple[str, str]],
        where each element is a tuple consisted with the local path to an object and its relative path to maintain the folder structure
    """
    files_to_upload = []
    for root, _, files in os.walk(folder_to_upload):
        for file in files:
            file_full_path = os.path.join(root, file)
            file_relative_path = os.path.relpath(file_full_path, folder_to_upload)
            files_to_upload.append((file_full_path, file_relative_path))
    return files_to_upload


def _get_local_path_to_download_objs(s3_objs: List[str], prefix: str, local_path: str) -> List[str]:
    """
    Get a list of paths to download s3 objects to.
    The paths will mirror the structure of the s3 folder (prefix)
    
    Parameters
    ----------
    s3_objs: List[str]
        List of objects needed to be downloaded.
        This list should be contents of a folder in s3
    prefix: str
        The prefix of the s3 folder to download
    local_path: str
        The local path to download contents to
        
    Returns
    -------
    List[str]
       The local paths of all the objects to be downloaded to
    """
    for obj in s3_objs:
        if obj.endswith('/'):
            raise ValueError(f'Folders are not supported: {obj}')
        elif obj == '':
            raise ValueError(f'Cannot have empty path for an s3 object!')

    # find the local path to download objs to
    local_obj_paths = [os.path.normpath(os.path.join(local_path, os.path.relpath(obj, prefix))) for obj in s3_objs]
    return local_obj_paths
