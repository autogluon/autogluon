import logging
import os
import shutil

from typing import List, Tuple, Optional

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
        

def download_s3_folder(
    bucket: str,
    prefix: str,
    local_path: str,
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
    if len(prefix) > 0:
        assert prefix.endswith("/"), "Please provide a prefix to a folder and end it with '/'"
    import boto3
    s3 = boto3.resource("s3")
    s3_bucket = s3.Bucket(bucket)
    # objs = list(bucket.objects.filter(Prefix=prefix))
    # objs = [obj.key for obj in objs]
    objs = list_bucket_prefix_suffix_contains_s3(bucket=bucket, prefix=prefix)
    local_obj_paths, folder_to_create = _get_local_path_to_download_objs_and_local_dir_to_create(
        s3_objs=objs,
        prefix=prefix,
        local_path=local_path
    )
    if verbose:
        logger.log(20, f"Will download {len(local_obj_paths)} objects from s3://{bucket}/{prefix} to {local_path}")
    if os.path.isdir(local_path) and not dry_run:
        if error_if_exists:
            raise ValueError(f"Directory {local_path} already exsists. Please pass in a different `local_path` or set `error_if_exsits` to `False`")
        if delete_if_exists:
            logger.warning(f"Will delete {local_path} and all its content within because this folder already exsists and `delete_if_exists` = `True`")
            shutil.rmtree(local_path)
    for folders in folder_to_create:
        if dry_run:
            logger.log(20, f"Will create directory {folders}")
        else:
            os.makedirs(folders, exist_ok=True)
    objs_no_folder = [obj for obj in objs if not obj.endswith("/")]  # remove directory so the download obj list can match local_obj_path
    for obj, local_obj_path in zip(objs_no_folder, local_obj_paths):
        if dry_run:
            logger.log(20, f"Will download {obj} to {local_obj_path}")
        else:
            s3_bucket.download_file(obj, local_obj_path)
            
            
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
            
def _get_local_path_to_download_objs_and_local_dir_to_create(s3_objs: List[str], prefix: str, local_path: str) -> Tuple[List[str], List[str]]:
    """
    Get a list of paths to download s3 objects to and a list of directories need to be created.
    The paths and directories will mirror the structure of the s3 folder (prefix)
    
    Parameters
    ----------
    s3_objs: List[str]
        List of objects including folders needs to be downloaded.
        This list should be contents of a folder in s3
    prefix: str
        The prefix of the s3 folder to download
    local_path: str
        The local path to download contents to
        
    Returns
    -------
    Tuple[List[str], List[str]]
        The first element is the local paths of all the objects to be downloaded to
        The second element is the folders that needs to be created
    """
    objs_no_folder = [obj for obj in s3_objs if not obj.endswith("/")]
    objs_folder_only = [obj for obj in s3_objs if obj.endswith("/")]
    # find the local path to download objs to
    local_obj_paths = [os.path.normpath(os.path.join(local_path, os.path.relpath(obj, prefix))) for obj in objs_no_folder]
    # add local path to folders
    folder_to_create = [os.path.normpath(os.path.join(local_path, os.path.relpath(folder, prefix))) for folder in objs_folder_only]
    folder_to_create += [os.path.dirname(os.path.normpath(path)) for path in local_obj_paths]
    folder_to_create = list(set(folder_to_create))
    folder_to_create = [folder for folder in folder_to_create if folder not in [".", ""]]  # current dir should be removed
    return local_obj_paths, folder_to_create
